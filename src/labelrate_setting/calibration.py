import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import numpy as np
import os
import gc
from pathlib import Path
from src.data.data_utils import load_base_data, load_node_to_nearest_training
from src.model.model import create_model
from src.calibrator.calibrator import \
    TS, VS, ETS, CaGCN, GATS, IRM, SplineCalib, Dirichlet, OrderInvariantCalib, RBS_cal
from src.calibloss import \
    NodewiseECE, NodewiseBrier, NodewiseNLL, Reliability, NodewiseKDE, \
    NodewiswClassECE
from src.utils import \
    set_global_seeds, arg_parse, name_model, create_nested_defaultdict, \
    metric_mean, metric_std, default_cal_wdecay, save_prediction
from src.calibrator import rbs



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
        self.brier_fn = NodewiseBrier(index)
        self.ece_fn = NodewiseECE(index, bins, scheme, norm)
        self.kde_fn = NodewiseKDE(index, norm)
        self.cls_ece_fn = NodewiswClassECE(index, bins, scheme, norm)

    def acc(self) -> float:
        preds = torch.argmax(self.logits, dim=1)[self.node_index]
        return torch.mean(
            (preds == self.gts[self.node_index]).to(torch.get_default_dtype())
        ).item()

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
    acc, nll, brier, ece, kde, cls_ece = eval.acc(), eval.nll(), \
                                eval.brier(), eval.ece(), eval.kde(), eval.cls_ece()
    eval_result.update({'acc':acc,
                        'nll':nll,
                        'bs':brier,
                        'ece':ece,
                        'kde':kde,
                        'cls_ece': cls_ece})
    reliability = eval.reliability()
    del eval
    gc.collect()
    return eval_result, reliability


def main(run, eval_type_list, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = create_nested_defaultdict(eval_type_list)
    cal_val_result = create_nested_defaultdict(eval_type_list)
    cal_test_result = create_nested_defaultdict(eval_type_list)
    
    dir = Path(os.path.join('model_labelrate/model', args.dataset, args.split_type, 'labelrate_'+str(args.labelrate)))
    model_name = args.model + "_run" + str(run)
    file_name = dir / (model_name + '.pt')
    checkpoint = torch.load(file_name)

    # Load data
    dataset = load_base_data(args.dataset)
    data = dataset.data.to(device)
    data.train_mask = checkpoint['train_mask']
    data.val_mask = checkpoint['val_mask']
    data.test_mask = checkpoint['test_mask'] 

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
    # elif args.calibration == 'RBS-cal':
    #     temp_model = RBS_cal(model, dataset, dataset.num_classes, args.num_bins_rbs)
    elif args.calibration == 'CaGCN':
        temp_model = CaGCN(model, data.num_nodes, dataset.num_classes, args.cal_dropout_rate)
    elif args.calibration == 'GATS':
        # dist_to_train = load_node_to_nearest_training(args.dataset, args.split_type, split, fold)
        dist_to_train = None
        temp_model = GATS(model, data.edge_index, data.num_nodes, data.train_mask,
                        dataset.num_classes, dist_to_train, args.gats_args)
    
    
    ### Train the calibrator on validation set and validate it on the training set
    cal_wdecay = args.cal_wdecay if args.cal_wdecay is not None else default_cal_wdecay(args)
    temp_model.fit(data, data.val_mask, data.train_mask, cal_wdecay)
    with torch.no_grad():
        temp_model.eval()
        logits = temp_model(data.x, data.edge_index)
        log_prob = F.log_softmax(logits, dim=1).detach()


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


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    seed = np.random.randint(0, 10000)
    print(f'the seed is {seed}')
    set_global_seeds(seed)
    eval_type_list = ['Nodewise']
    num_runs = 10

    uncal_test_total = create_nested_defaultdict(eval_type_list)
    cal_val_total = create_nested_defaultdict(eval_type_list)
    cal_test_total = create_nested_defaultdict(eval_type_list)
    reliabilities_uncal, reliabilities_cal = [], []
    print(f'-----------------------------------calibration with {args.model} on {args.dataset} - labelrate{args.labelrate}-----------------------------------')

    for run in range(num_runs):

        (uncal_test_result,
            cal_val_result,
            cal_test_result,
            reliability_uncal,
            reliability_cal) = main(run, eval_type_list, args)
        
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

    # print results
    for name, result in zip(['Uncal', args.calibration], [uncal_test_total, cal_test_total]):
        print(name)
        for eval_type in eval_type_list:
            test_mean = metric_mean(result[eval_type])
            test_std = metric_std(result[eval_type])
            print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
                                f"NLL: &{test_mean['nll']:.4f}$\pm${test_std['nll']:.4f} \t" + \
                                f"Brier: &{test_mean['bs']:.4f}$\pm${test_std['bs']:.4f} \t" + \
                                f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f} \t" + \
                                f"Classwise-ECE: &{test_mean['cls_ece']:.2f}$\pm${test_std['cls_ece']:.2f} \t" + \
                                f"KDE: &{test_mean['kde']:.2f}$\pm${test_std['kde']:.2f}")
