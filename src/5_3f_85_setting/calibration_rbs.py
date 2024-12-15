import os
import numpy as np
import random
import abc
import gc
import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import os
import gc
from pathlib import Path
from src.data.data_utils import load_data, load_node_to_nearest_training
from src.model.model import create_model
from src.calibloss import \
    NodewiseECE, NodewiseBrier, NodewiseNLL, Reliability, NodewiseKDE, \
    NodewiswClassECE
from src.utils import \
    set_global_seeds, arg_parse, name_model, create_nested_defaultdict, \
    metric_mean, metric_std, default_cal_wdecay, save_prediction

from calGNN.create_model import create_gcn_model, create_gat_model
from calGNN.main_loss import train_gnn, calibrate_gnn
from calGNN.args import get_args
import argparse
from calGNN.main import calibrate_gnn_rbs

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def metric_mean_rbs(result_dict):
    out = {}
    for key, val in result_dict.items():
        if key in ['acc', 'ece', 'marg_ece', 'test_acc']:
            weight = 100
        else:
            weight = 1
        out[key] = np.mean(val) * weight
    return out

def metric_std_rbs(result_dict):
    out = {}
    for key, val in result_dict.items():
        if key in ['acc', 'ece', 'marg_ece', 'test_acc']:
            weight = 100
        else:
            weight = 1
        out[key] = np.sqrt(np.var(val)) * weight
    return out

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


def eval_test(data, logits, mask_name=None):

    if mask_name == 'Train':
        mask = data.train_mask
        label = data.y
    elif mask_name == 'Val':
        mask = data.val_mask
        label = data.y
    elif mask_name == 'Test':
        mask = data.test_mask
        label = data.y
    elif mask_name == 'Train-logits':
        mask = torch.ones(logits.shape[0], dtype=torch.bool).to('cpu')
        label = data.y[data.train_mask]
    elif mask_name == 'Test-logits':
        mask = torch.ones(logits.shape[0], dtype=torch.bool).to('cpu')
        label = data.y[data.test_mask].to('cpu')
    eval_result = {}
    
    eval = NodewiseMetrics(logits, label, mask)
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


def main(split, init, eval_type_list, uncal_vanllia_result, test_vanllia_result, args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = create_nested_defaultdict(eval_type_list)
    cal_val_result = create_nested_defaultdict(eval_type_list)
    cal_test_result = create_nested_defaultdict(eval_type_list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):

        # Load data
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data.to(device)

        # Load model
        model_name = name_model(fold, args)
        model = create_model(dataset, args).to(device)
        dir = Path(os.path.join('model_latest', args.dataset, args.split_type, 'split'+str(split), 'init'+ str(init)))
        file_name = dir / (model_name + '.pt')
        model.load_state_dict(torch.load(file_name))
        torch.cuda.empty_cache()

        # if args.model == "GCN":
        #     model, optimizer = create_gcn_model(dataset)
        # elif args.model == "GAT":
        #     model, optimizer = create_gat_model(dataset)
        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index)
            log_prob = F.log_softmax(logits, dim=1).detach()

        ### Store uncalibrated result
        for eval_type in eval_type_list:
            eval_result, reliability = eval_test(data, logits, 'Test')
            for metric in eval_result:
                uncal_test_result[eval_type][metric].append(eval_result[metric])
        torch.cuda.empty_cache()

        ### Calibration RBS

        # print("Calibrate...")
        cal_logits_test, vanllia_uncal, vanllia_cal = calibrate_gnn_rbs(dataset, model, args)
        for ind, key in enumerate(uncal_vanllia_result.keys()):
            uncal_vanllia_result[key].append(vanllia_uncal[ind])
            test_vanllia_result[key].append(vanllia_cal[ind])
        ### The training set is the validation set for the calibrator

        # for eval_type in eval_type_list:
        #     eval_result, _ = eval_test(data, log_prob, 'Train-logits')
        #     for metric in eval_result:
        #         cal_val_result[eval_type][metric].append(eval_result[metric])
            
        for eval_type in eval_type_list:
            eval_result, reliability = eval_test(data, cal_logits_test, 'Test-logits')
            for metric in eval_result:
                cal_test_result[eval_type][metric].append(eval_result[metric])
        torch.cuda.empty_cache()

    return uncal_test_result, cal_test_result
        
        


if __name__ == '__main__':
    args = arg_parse(get_args)
    print(args)
    print(f'----------------num_bins: {args.num_bins_rbs}--------------------')
    set_global_seeds(args.seed)
    max_splits,  max_init = 5, 5
    eval_type_list = ['Nodewise']
    uncal_test_total = create_nested_defaultdict(eval_type_list)
    cal_val_total = create_nested_defaultdict(eval_type_list)
    cal_test_total = create_nested_defaultdict(eval_type_list)

    uncal_vanllia_result = {'ece':[], 'marg_ece':[], 'nll':[], 'test_acc':[]}
    test_vanllia_result = {'ece':[], 'marg_ece':[], 'nll':[], 'test_acc':[]}
    for split in range(max_splits):
        for init in range(max_init):
            print(split, init)
            (uncal_test_result,
             cal_test_result) = main(split, init, eval_type_list, uncal_vanllia_result, test_vanllia_result, args)
            for eval_type, eval_metric in uncal_test_result.items():
                for metric in eval_metric:
                    uncal_test_total[eval_type][metric].extend(uncal_test_result[eval_type][metric])
                    cal_test_total[eval_type][metric].extend(cal_test_result[eval_type][metric])
    print('------vanllia calGNN metric result------------')
    test_mean_val = metric_mean_rbs(uncal_vanllia_result)
    test_std_val = metric_std_rbs(uncal_vanllia_result)
    print(f"uncal vanllia Accuracy: &{test_mean_val['test_acc']:.2f}$\pm${test_std_val['test_acc']:.2f} \t" + \
                        f"NLL: &{test_mean_val['nll']:.4f}$\pm${test_std_val['nll']:.4f} \t" + \
                        f"ECE: &{test_mean_val['ece']:.2f}$\pm${test_std_val['ece']:.2f} \t" + \
                        f"Marg-ECE: &{test_mean_val['marg_ece']:.2f}$\pm${test_std_val['marg_ece']:.2f} \t")


    test_mean_val = metric_mean_rbs(test_vanllia_result)
    test_std_val = metric_std_rbs(test_vanllia_result)
    print(f"cal vanllia Accuracy: &{test_mean_val['test_acc']:.2f}$\pm${test_std_val['test_acc']:.2f} \t" + \
                        f"NLL: &{test_mean_val['nll']:.4f}$\pm${test_std_val['nll']:.4f} \t" + \
                        f"ECE: &{test_mean_val['ece']:.2f}$\pm${test_std_val['ece']:.2f} \t" + \
                        f"Marg-ECE: &{test_mean_val['marg_ece']:.2f}$\pm${test_std_val['marg_ece']:.2f} \t")

    print('------GATS metric result----------')
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

