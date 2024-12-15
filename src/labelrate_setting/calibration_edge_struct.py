import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import numpy as np
import copy
import os
import gc
from pathlib import Path
from src.data.data_utils import load_base_data, load_node_to_nearest_training
from src.model.model import create_model
from src.calibrator.calibrator import \
    TS, VS, ETS, CaGCN, GATS, IRM, SplineCalib, Dirichlet, OrderInvariantCalib, Edge_Weight
from src.calibloss import \
    NodewiseECE, NodewiseBrier, NodewiseNLL, Reliability, NodewiseKDE, \
    NodewiswClassECE, NodewiseConf
from src.utils import \
    set_global_seeds, arg_parse, name_model, create_nested_defaultdict, \
    metric_mean, metric_std, default_cal_wdecay, save_prediction

from src.utils import draw_del_ratio_ece, plot_reliabilities

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# collects metrics for evaluation
class Metrics(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def acc(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def nll(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def conf(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def brier(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def ece(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def reliability(self) -> Reliability:
        raise NotImplementedError

    @abc.abstractmethod
    def kde(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def cls_ece(self) -> float:
        raise NotImplementedError

class NodewiseMetrics(Metrics):
    def __init__(
            self, logits: Tensor, gts: LongTensor, index: LongTensor,
            bins: int = 15, scheme: str = 'equal_width', norm=1):
        self.node_index = index
        self.logits = logits
        self.gts = gts
        self.nll_fn = NodewiseNLL(index)
        self.conf_fn = NodewiseConf(index)
        self.brier_fn = NodewiseBrier(index)
        self.ece_fn = NodewiseECE(index, bins, scheme, norm)
        self.kde_fn = NodewiseKDE(index, norm)
        self.cls_ece_fn = NodewiswClassECE(index, bins, scheme, norm)

    def acc(self) -> float:
        preds = torch.argmax(self.logits, dim=1)[self.node_index]
        return torch.mean(
            (preds == self.gts[self.node_index]).to(torch.get_default_dtype())
        ).item()

    def conf(self) -> float:
        return self.conf_fn(self.logits, self.gts).item()

    def nll(self) -> float:
        return self.nll_fn(self.logits, self.gts).item()

    def brier(self) -> float:
        return self.brier_fn(self.logits, self.gts).item()

    def ece(self) -> float:
        return self.ece_fn(self.logits, self.gts).item()

    def reliability(self) -> Reliability:
        return self.ece_fn.get_reliability(self.logits, self.gts)
    
    def kde(self) -> float:
        return self.kde_fn(self.logits, self.gts).item()

    def cls_ece(self) -> float:
        return self.cls_ece_fn(self.logits, self.gts).item()


def eval(data, logits, mask_name):
    if mask_name == 'Train':
        mask = data.train_mask
    elif mask_name == 'Val':
        mask = data.val_mask
    elif mask_name == 'Test':
        mask = data.test_mask
    else:
        raise ValueError("Invalid mask_name")
    eval_result = {}
    eval = NodewiseMetrics(logits, data.y, mask)
    acc, nll, brier, ece, kde, cls_ece, conf = eval.acc(), eval.nll(), \
                                eval.brier(), eval.ece(), eval.kde(), eval.cls_ece(), eval.conf()
    eval_result.update({'acc':acc,
                        'nll':nll,
                        'conf':conf,
                        'bs':brier,
                        'ece':ece,
                        'kde':kde,
                        'cls_ece': cls_ece})
    reliability = eval.reliability()
    del eval
    gc.collect()
    return eval_result, reliability


def main(run, eval_type_list, ratio, ratio_name, is_delete, is_add, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = create_nested_defaultdict(eval_type_list)
    dcgc_test_result = create_nested_defaultdict(eval_type_list)
    cal_val_result = create_nested_defaultdict(eval_type_list)
    cal_test_result = create_nested_defaultdict(eval_type_list)
    cal_test_result_nowt = create_nested_defaultdict(eval_type_list)

    # ratio_name = 'delete_'+str(ratio) if is_delete else 'add_'+str(ratio)
    dir = Path(os.path.join('model_labelrate/model_modify_edge_all', args.dataset, 'labelrate_'+str(args.labelrate), ratio_name))
    model_name = args.model + "_run" + str(run)
    file_name = dir / (model_name + '.pt')
    checkpoint = torch.load(file_name)

    # Load data
    dataset = load_base_data(args.dataset)
    data = dataset.data.to(device)
    data.train_mask = checkpoint['train_mask']
    data.val_mask = checkpoint['val_mask']
    data.test_mask = checkpoint['test_mask'] 

    # modify data
    if is_delete:
        new_edge_index = checkpoint['new_edge_index']
        print(f'the modify-delete {args.dataset} edges: {new_edge_index.size(1)} with ratio {ratio}') 
        data.edge_index = new_edge_index
        
    elif is_add:   
        new_edge_index =  checkpoint['new_edge_index']
        print(f'the modeify-add {args.dataset} edges: {new_edge_index.size(1)} with num {ratio}')
        data.edge_index = new_edge_index
    
    # Load model
    model = create_model(dataset, args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.cuda.empty_cache()

    with torch.no_grad():
        model.eval()
        logits = model(data.x, data.edge_index)
        log_prob = F.log_softmax(logits, dim=1).detach()

    for eval_type in eval_type_list:
        eval_result, reliability_uncal = eval(data, logits, 'Test')
        for metric in eval_result:
            uncal_test_result[eval_type][metric].append(eval_result[metric])
    torch.cuda.empty_cache()

    # GCDC
    cal_wdecay = args.cal_wdecay if args.cal_wdecay is not None else default_cal_wdecay(args)
    # Train the calibrator on validation set and validate it on the training set
    ew = Edge_Weight(model, dataset.num_classes, args.dcgc_dropout).to(device)
    ew.fit(data, data.val_mask, data.train_mask, wdecay=cal_wdecay)
    with torch.no_grad():
        ew.eval()
        logits = ew(data.x, data.edge_index)
        prob = F.softmax(logits, dim=1)
        edge_weight = ew.get_weight(data.x, data.edge_index)
    # dcgc_beta = [0.1, 0.2, 1, 10]
    # dcgc_alpha = [0.1, 0.3, 0.5, 1]
    pred = torch.exp(args.dcgc_beta * prob)
    pred /= torch.sum(pred, dim=1, keepdim=True)

    col, row = data.edge_index
    coefficient = torch.norm(pred[col] - pred[row], dim=1)
    coefficient = 1 / (coefficient + args.dcgc_alpha)
    
    edge_weight = edge_weight.reshape(-1)
    edge_weight = edge_weight * coefficient
    edge_weight = edge_weight.reshape([data.num_edges, 1])

    with torch.no_grad():
        ew.eval()
        logits = ew(data.x, data.edge_index, edge_weight)

    for eval_type in eval_type_list:
        eval_result, reliability_dcgc = eval(data, logits, 'Test')
        for metric in eval_result:
            dcgc_test_result[eval_type][metric].append(eval_result[metric])
    torch.cuda.empty_cache()

    ### Calibration
    if args.calibration == 'TS':
        temp_model = TS(model)
    elif args.calibration == 'IRM':
        temp_model = IRM(model)
    elif args.calibration == 'Spline':
        temp_model = SplineCalib(model, 7)
    elif args.calibration == 'Dirichlet':
        temp_model = Dirichlet(model, dataset.num_classes)
    elif args.calibration == 'OrderInvariant':
        temp_model = OrderInvariantCalib(model, dataset.num_classes)
    elif args.calibration == 'VS':
        temp_model = VS(model, dataset.num_classes)
    elif args.calibration == 'ETS':
        temp_model = ETS(model, dataset.num_classes)
    elif args.calibration == 'CaGCN':
        temp_model = CaGCN(model, data.num_nodes, dataset.num_classes, args.cal_dropout_rate)
    elif args.calibration == 'GATS':
        # dist_to_train = load_node_to_nearest_training(args.dataset, args.split_type, split, fold)
        dist_to_train = None
        temp_model = GATS(model, data.edge_index, data.num_nodes, data.train_mask,
                        dataset.num_classes, dist_to_train, args.gats_args)
    
    
    ### Train the calibrator on validation set and validate it on the training set
    cal_wdecay = args.cal_wdecay if args.cal_wdecay is not None else default_cal_wdecay(args)

    temp_model_nowt = copy.deepcopy(temp_model)
    temp_model_nowt.fit(data, data.val_mask, data.train_mask, cal_wdecay, edge_weight=None)
    temp_model.fit(data, data.val_mask, data.train_mask, cal_wdecay, edge_weight=edge_weight)
    with torch.no_grad():
        temp_model_nowt.eval()
        logits_nowt = temp_model_nowt(data.x, data.edge_index)
        temp_model.eval()
        logits = temp_model(data.x, data.edge_index, edge_weight)

    ### The training set is the validation set for the calibrator
    for eval_type in eval_type_list:
        eval_result, _ = eval(data, logits, 'Train')
        for metric in eval_result:
            cal_val_result[eval_type][metric].append(eval_result[metric])

    for eval_type in eval_type_list:
        eval_result_nowt, reliability_cal_nowt = eval(data, logits_nowt, 'Test')
        eval_result, reliability_cal = eval(data, logits, 'Test')
        for metric in eval_result:
            cal_test_result_nowt[eval_type][metric].append(eval_result_nowt[metric])
            cal_test_result[eval_type][metric].append(eval_result[metric])
    torch.cuda.empty_cache()
    return uncal_test_result, dcgc_test_result, cal_val_result, cal_test_result_nowt, cal_test_result, reliability_uncal, reliability_dcgc, reliability_cal_nowt, reliability_cal

def per_ratio_ece(ratio, num_runs, is_delete=False, is_add=False, ratio_name=None, is_save_calfig=False):

    uncal_test_total = create_nested_defaultdict(eval_type_list)
    dcgc_test_total = create_nested_defaultdict(eval_type_list)
    cal_val_total = create_nested_defaultdict(eval_type_list)
    cal_test_total_nowt = create_nested_defaultdict(eval_type_list)
    cal_test_total = create_nested_defaultdict(eval_type_list)
    reliabilities_uncal, reliabilities_dcgc, reliabilities_cal_nowt, reliabilities_cal = [], [], [], []

    for run in range(num_runs):

        (uncal_test_result,
            dcgc_test_result,
            cal_val_result,
            cal_test_result_nowt,
            cal_test_result,
            reliability_uncal,
            reliability_dcgc,
            reliability_cal_nowt,
            reliability_cal) = main(run, eval_type_list, ratio, ratio_name, is_delete, is_add, args)
        
        reliabilities_uncal.append(reliability_uncal)
        reliabilities_dcgc.append(reliability_dcgc)
        reliabilities_cal_nowt.append(reliability_cal_nowt)
        reliabilities_cal.append(reliability_cal)

        for eval_type, eval_metric in uncal_test_result.items():
            for metric in eval_metric:
                uncal_test_total[eval_type][metric].extend(uncal_test_result[eval_type][metric])
                dcgc_test_total[eval_type][metric].extend(dcgc_test_result[eval_type][metric])
                cal_val_total[eval_type][metric].extend(cal_val_result[eval_type][metric])
                cal_test_total_nowt[eval_type][metric].extend(cal_test_result_nowt[eval_type][metric])
                cal_test_total[eval_type][metric].extend(cal_test_result[eval_type][metric])

    val_mean = metric_mean(cal_val_total['Nodewise'])
    # validate calibrator
    print(f"Val NNL: &{val_mean['nll']:.4f}")
    test_mean_ece, test_mean_acc = [], []

    # print results
    for name, result, reliability in zip(['Uncal', 'DCGC', args.calibration, 'DCGC+'+args.calibration], [uncal_test_total, dcgc_test_total, cal_test_total_nowt, cal_test_total], [reliabilities_uncal, reliabilities_dcgc, reliabilities_cal_nowt, reliabilities_cal]):
        print(name)
        for eval_type in eval_type_list:
            test_mean = metric_mean(result[eval_type])
            test_std = metric_std(result[eval_type])
            print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
                                f"NLL: &{test_mean['nll']:.4f}$\pm${test_std['nll']:.4f} \t" + \
                                f"Conf: &{test_mean['conf']:.2f}$\pm${test_std['conf']:.2f} \t" + \
                                f"Brier: &{test_mean['bs']:.4f}$\pm${test_std['bs']:.4f} \t" + \
                                f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f} \t" + \
                                f"Classwise-ECE: &{test_mean['cls_ece']:.2f}$\pm${test_std['cls_ece']:.2f} \t" + \
                                f"KDE: &{test_mean['kde']:.2f}$\pm${test_std['kde']:.2f}")
        # test_mean['ece'] with ratio
        test_mean_ece.append(test_mean['ece'])
        test_mean_acc.append(test_mean['acc'])

        # save the calibration figure
        if is_save_calfig:
            parent_fold = os.path.join('figure_labelrate/confidence_acc/modify_edge_cagcn/', args.dataset,'labelrate_'+str(args.labelrate), args.model, name)
            dir = Path(parent_fold)
            dir.mkdir(parents=True, exist_ok=True)
            save_pth_fig = dir / (ratio_name + '.png')
            plot_reliabilities(reliability, title=args.dataset, saveto=save_pth_fig)  
    
    return test_mean_ece, test_mean_acc





if __name__ == '__main__':
    args = arg_parse()
    print(args)
    # seed = np.random.randint(0, 10000)
    # print(f'the seed is {seed}')
    set_global_seeds(args.seed)
    eval_type_list = ['Nodewise']
    num_runs = 10
    delete_ratio_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    add_num_list = [0, 20, 50, 80, 110, 140, 170, 200, 230, 260, 300]
    
    delete_edge_ece_uncal, delete_edge_ece_dcgc, delete_edge_ece_cal_nowt, delete_edge_ece_cal = [], [], [], []
    delete_edge_acc_uncal, delete_edge_acc_dcgc, delete_edge_acc_cal_nowt, delete_edge_acc_cal = [], [], [], []
    add_edge_ece_uncal, add_edge_ece_dcgc, add_edge_ece_cal_nowt, add_edge_ece_cal = [], [], [], []
    add_edge_acc_uncal, add_edge_acc_dcgc, add_edge_acc_cal_nowt, add_edge_acc_cal = [], [], [], []
    print(f'-----------------------------------training with {args.model} on {args.dataset}-----------------------------------')
    if args.is_edge_delete:
        for delete_ratio in delete_ratio_list:
            ratio_name = 'delete_'+ str(delete_ratio)
            de_ece, de_acc = per_ratio_ece(delete_ratio, num_runs, is_delete=True, ratio_name=ratio_name, is_save_calfig=True)
            delete_edge_ece_uncal.append(de_ece[0])
            delete_edge_ece_dcgc.append(de_ece[1])
            delete_edge_ece_cal_nowt.append(de_ece[2])
            delete_edge_ece_cal.append(de_ece[3])

            delete_edge_acc_uncal.append(de_acc[0])
            delete_edge_acc_dcgc.append(de_acc[1])
            delete_edge_acc_cal_nowt.append(de_acc[2])
            delete_edge_acc_cal.append(de_acc[3])

        print(f'-------------------------Delete edge ratio {delete_ratio}---------------------------------------------------')
        print(f'delete edge mean ece Uncal : {delete_edge_ece_uncal}')
        print(f'delete edge mean ece DCGC  : {delete_edge_ece_dcgc}')
        print(f'delete edge mean ece {args.calibration} : {delete_edge_ece_cal_nowt}')
        print(f'delete edge mean ece DCGC+{args.calibration} : {delete_edge_ece_cal}')

        print(f'delete edge mean acc Uncal : {delete_edge_acc_uncal}')
        print(f'delete edge mean acc DCGC  : {delete_edge_acc_dcgc}')
        print(f'delete edge mean acc {args.calibration} : {delete_edge_acc_cal_nowt}')
        print(f'delete edge mean acc DCGC+{args.calibration} : {delete_edge_acc_cal}')
        print('----------------------------------------------------------------------------')

        # draw_del_ratio_ece(delete_ratio_list, delete_edge_ece_uncal, delete_edge_ece_cal, title='delete-Edge ECE Comparison', save_path='/root/GATS/figure/del_edge_ece.png')

    elif args.is_edge_add:
        for add_num in add_num_list:
            ratio_name = 'add_' + str(add_num)
            add_ece, add_acc = per_ratio_ece(add_num, num_runs, is_add=True, ratio_name=ratio_name, is_save_calfig=True)
            print(add_ece)
            add_edge_ece_uncal.append(add_ece[0])
            add_edge_ece_dcgc.append(add_ece[1])
            add_edge_ece_cal_nowt.append(add_ece[2])
            add_edge_ece_cal.append(add_ece[3])

            add_edge_acc_uncal.append(add_acc[0])
            add_edge_acc_dcgc.append(add_acc[1])
            add_edge_acc_cal_nowt.append(add_acc[2])
            add_edge_acc_cal.append(add_acc[3])
        print(f'---------------------------Add edge num {add_num}-------------------------------------------------')
        print(f'add edge mean ece Uncal : {add_edge_ece_uncal}')
        print(f'add edge mean ece DCGC  : {add_edge_ece_dcgc}')
        print(f'add edge mean ece {args.calibration} : {add_edge_ece_cal_nowt}')
        print(f'add edge mean ece DCGC+{args.calibration} : {add_edge_ece_cal}')

        print(f'add edge mean acc Uncal : {add_edge_acc_uncal}')
        print(f'add edge mean acc DCGC  : {add_edge_acc_dcgc}')
        print(f'add edge mean acc {args.calibration} : {add_edge_acc_cal_nowt}')
        print(f'add edge mean acc DCGC+{args.calibration} : {add_edge_acc_cal}')
        # draw_del_ratio_ece(add_ratio_list, add_edge_ece_uncal, add_edge_ece_cal, title='Add-Edge ECE Comparison', save_path='/root/GATS/figure/add_edge_ece.png')
        print('----------------------------------------------------------------------------')

    