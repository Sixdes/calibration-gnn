import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import os
import gc
from pathlib import Path
from src.data.data_utils import load_data, load_node_to_nearest_training
from src.model.model import create_model
from src.calibrator.calibrator import \
    TS, VS, ETS, CaGCN, GATS, IRM, SplineCalib, Dirichlet, OrderInvariantCalib, RBS_cal
from src.calibloss import \
    NodewiseECE, NodewiseBrier, NodewiseNLL, Reliability, NodewiseKDE, \
    NodewiswClassECE, NodewiseConf
from src.utils import \
    set_global_seeds, arg_parse, name_model, create_nested_defaultdict, \
    metric_mean, metric_std, default_cal_wdecay, save_prediction
from src.calibrator import rbs

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


def main(split, init, eval_type_list, ratio, ratio_name, is_delete, is_add, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = create_nested_defaultdict(eval_type_list)
    cal_val_result = create_nested_defaultdict(eval_type_list)
    cal_test_result = create_nested_defaultdict(eval_type_list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):

        dir = Path(os.path.join('backup/model_modify_edge_all', args.dataset, ratio_name, 'split'+str(split), 'init'+ str(init)))
        model_name = name_model(fold, args)
        file_name = dir / (model_name + '.pt')
        checkpoint = torch.load(file_name)

        # Load data
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data.to(device)

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

        for eval_type in eval_type_list:
            eval_result, reliability_uncal = eval(data, logits, 'Test')
            for metric in eval_result:
                uncal_test_result[eval_type][metric].append(eval_result[metric])
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
            dist_to_train = load_node_to_nearest_training(args.dataset, args.split_type, split, fold)
            temp_model = GATS(model, data.edge_index, data.num_nodes, data.train_mask,
                            dataset.num_classes, dist_to_train, args.gats_args)
        
        
        ### Train the calibrator on validation set and validate it on the training set
        cal_wdecay = args.cal_wdecay if args.cal_wdecay is not None else default_cal_wdecay(args)
        temp_model.fit(data, data.val_mask, data.train_mask, cal_wdecay)
        with torch.no_grad():
            temp_model.eval()
            logits = temp_model(data.x, data.edge_index)

        ### The training set is the validation set for the calibrator
        for eval_type in eval_type_list:
            eval_result, _ = eval(data, logits, 'Train')
            for metric in eval_result:
                cal_val_result[eval_type][metric].append(eval_result[metric])

        for eval_type in eval_type_list:
            eval_result, reliability_cal = eval(data, logits, 'Test')
            for metric in eval_result:
                cal_test_result[eval_type][metric].append(eval_result[metric])
        torch.cuda.empty_cache()
    return uncal_test_result, cal_val_result, cal_test_result, reliability_uncal, reliability_cal

def per_ratio_ece(ratio, max_splits, max_init, is_delete=False, is_add=False, ratio_name=None, is_save_calfig=False):

    uncal_test_total = create_nested_defaultdict(eval_type_list)
    cal_val_total = create_nested_defaultdict(eval_type_list)
    cal_test_total = create_nested_defaultdict(eval_type_list)
    reliabilities_uncal, reliabilities_cal = [], []
    
    for split in range(max_splits):
        for init in range(max_init):
            (uncal_test_result,
             cal_val_result,
             cal_test_result,
             reliability_uncal,
             reliability_cal) = main(split, init, eval_type_list, ratio, ratio_name, is_delete, is_add, args)
            
            reliabilities_uncal.append(reliability_uncal)
            reliabilities_cal.append(reliability_cal)

            for eval_type, eval_metric in uncal_test_result.items():
                for metric in eval_metric:
                    uncal_test_total[eval_type][metric].extend(uncal_test_result[eval_type][metric])
                    cal_val_total[eval_type][metric].extend(cal_val_result[eval_type][metric])
                    cal_test_total[eval_type][metric].extend(cal_test_result[eval_type][metric])

    val_mean = metric_mean(cal_val_total['Nodewise'])
    # validate calibrator
    print(f"Val NNL: &{val_mean['nll']:.4f}")
    test_mean_record = {'ece':[], 'acc':[], 'conf':[]}

    # print results
    for name, result, reliability in zip(['Uncal', args.calibration], [uncal_test_total, cal_test_total], [reliabilities_uncal, reliabilities_cal]):
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
        test_mean_record['ece'].append(test_mean['ece'])
        test_mean_record['acc'].append(test_mean['acc'])
        test_mean_record['conf'].append(test_mean['conf'])

        # save the calibration figure
        if is_save_calfig:
            parent_fold = os.path.join('figure_5_3f_85/confidence_acc/modify_edge/', args.dataset, args.model, name)
            dir = Path(parent_fold)
            dir.mkdir(parents=True, exist_ok=True)
            save_pth_fig = dir / (ratio_name + '.png')
            print(save_pth_fig)
            plot_reliabilities(reliability, title=args.dataset, saveto=save_pth_fig)  
    
    return test_mean_record





if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    eval_type_list = ['Nodewise']
    max_splits,  max_init = 3, 2
    delete_ratio_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    add_num_list = [0, 20, 50, 80, 110, 140, 170, 200, 230, 260, 300]
    
    delete_record_uncal = {'ece':[], 'acc':[], 'conf':[]}
    delete_record_cal = {'ece':[], 'acc':[], 'conf':[]}
    add_record_uncal = {'ece':[], 'acc':[], 'conf':[]}
    add_record_cal = {'ece':[], 'acc':[], 'conf':[]}

    print(f'-----------------------------------training with {args.model} on {args.dataset}-----------------------------------')
    if args.is_edge_delete:
        for delete_ratio in delete_ratio_list:
            ratio_name = 'delete_'+ str(delete_ratio)
            de_record = per_ratio_ece(delete_ratio, max_splits, max_init, is_delete=True, ratio_name=ratio_name, is_save_calfig=False)
            for metric in de_record.keys():
                delete_record_uncal[metric].append(de_record[metric][0])
                delete_record_cal[metric].append(de_record[metric][1])
            
        print('----------------------------------------------------------------------------')
        print(f'delete edge mean ece uncal : {delete_record_uncal["ece"]}')
        print(f'delete edge mean ece {args.calibration} : {delete_record_cal["ece"]}')
        print(f'delete edge mean acc uncal : {delete_record_uncal["acc"]}')
        print(f'delete edge mean acc {args.calibration} : {delete_record_cal["acc"]}')
        print(f'delete edge mean conf uncal : {delete_record_uncal["conf"]}')
        print(f'delete edge mean conf {args.calibration} : {delete_record_cal["conf"]}')
        print('----------------------------------------------------------------------------')

        # draw_del_ratio_ece(delete_ratio_list, delete_edge_ece_uncal, delete_edge_ece_cal, title='delete-Edge ECE Comparison', save_path='/root/GATS/figure/del_edge_ece.png')

    elif args.is_edge_add:
        for add_num in add_num_list:
            ratio_name = 'add_' + str(add_num)
            add_record = per_ratio_ece(add_num, max_splits, max_init, is_add=True, ratio_name=ratio_name, is_save_calfig=False)
            for metric in add_record.keys():
                add_record_uncal[metric].append(add_record[metric][0])
                add_record_cal[metric].append(add_record[metric][1])
        print('----------------------------------------------------------------------------')
        print(f'add edge mean ece uncal : {add_record_uncal["ece"]}')
        print(f'add edge mean ece {args.calibration} : {add_record_cal["ece"]}')
        print(f'add edge mean acc uncal : {add_record_uncal["acc"]}')
        print(f'add edge mean acc {args.calibration} : {add_record_cal["acc"]}')
        print(f'add edge mean conf uncal : {add_record_uncal["conf"]}')
        print(f'add edge mean conf {args.calibration} : {add_record_cal["conf"]}')
        # draw_del_ratio_ece(add_ratio_list, add_edge_ece_uncal, add_edge_ece_cal, title='Add-Edge ECE Comparison', save_path='/root/GATS/figure/add_edge_ece.png')
        print('----------------------------------------------------------------------------')

    