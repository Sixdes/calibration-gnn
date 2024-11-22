import abc
import argparse
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
    NodewiswClassECE, ECE
from src.utils import \
    set_global_seeds, name_model, create_nested_defaultdict, \
    metric_mean, metric_std, default_cal_wdecay, save_prediction
from src.calibrator import rbs
from src.utils import plot_reliabilities


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


def main(split, init, eval_type_list, parent_pth, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = create_nested_defaultdict(eval_type_list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))
    reliabilities = []

    for fold in range(max_fold):
        # Load data
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data.to(device)

        # Load model
        model = create_model(dataset, args).to(device)
        model_name = name_model(fold, args)
        dir = Path(os.path.join(parent_pth, args.dataset, args.split_type, 'split'+str(split), 'init'+ str(init)))
        file_name = dir / (model_name + '.pt')
        model.load_state_dict(torch.load(file_name))
        torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index)
            log_prob = F.log_softmax(logits, dim=1).detach()

        ### Store uncalibrated result
        if args.save_prediction:
            save_prediction(log_prob.cpu().numpy(), args.dataset, args.split_type, split, init, fold, args.model, "uncal")

        for eval_type in eval_type_list:
            eval_result, reliability = eval(data, logits, 'Test')
            reliabilities.append(reliability)
            for metric in eval_result:
                uncal_test_result[eval_type][metric].append(eval_result[metric])
        torch.cuda.empty_cache()

    return uncal_test_result, reliabilities

def arg_parse():
    parser = argparse.ArgumentParser(description='train.py and calibration.py share the same arguments')
    parser.add_argument('--seed', type=int, default=10, help='Random Seed')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora','Citeseer', 'Pubmed', 
                        'Computers', 'Photo', 'CS', 'Physics', 'CoraFull'])
    parser.add_argument('--split_type', type=str, default='5_3f_85', help='k-fold and test split')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT'])
    parser.add_argument('--verbose', action='store_true', default=False, help='Show training and validation loss')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='Weight decay for training phase')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate. 1.0 denotes drop all the weights to zero')
    
    parser.add_argument('--folds', type=int, default=3, help='K folds cross-validation for calibration')
    parser.add_argument('--ece-bins', type=int, default=15, help='number of bins for ece')
    parser.add_argument('--ece-scheme', type=str, default='equal_width', choices=ECE.binning_schemes, help='binning scheme for ece')
    parser.add_argument('--ece-norm', type=float, default=1.0, help='norm for ece')
    parser.add_argument('--save_prediction', action='store_true', default=False)
    
    parser.add_argument('--model_retrain_name', type=str, default='GCL',  help='model name')
    # CRL
    parser.add_argument("--alpha", type=float, default=0.98)
    parser.add_argument("--lmbda", type=float, default=1.0)
    # Logit_norm
    parser.add_argument('--temp_eff', type=float, default=1.)
    # GCL
    parser.add_argument('--gamma', type=float, default=0.01)
    # vis
    parser.add_argument('--is_draw_reliability', action='store_true', default=False, help='Draw Acc-Confidence Fig')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    eval_type_list = ['Nodewise']
    max_splits,  max_init = 5, 5
    
    uncal_test_total = create_nested_defaultdict(eval_type_list)
    print(f"---------The {args.model_retrain_name}-{args.model} from the {args.dataset} is being evaluted--------- ")
    parent_dir = 'model_'+args.model_retrain_name.lower()
    print('Load model from ', parent_dir)

    for split in range(max_splits):
        for init in range(max_init):
            # print(split, init)
            uncal_test_result, reliabilities = main(split, init, eval_type_list, parent_dir, args)
    
    if args.is_draw_reliability:
        if args.model_retrain_name == 'latest':
            title = args.dataset+'_'+args.model
        else:
            title = args.dataset+'_'+args.model_retrain_name
        save_pth = '/root/GATS/figure/'+title+'.png'
        print(save_pth)
        plot_reliabilities(reliabilities, title=args.dataset, saveto=save_pth)
    
    for split in range(max_splits):
        for init in range(max_init):
            # print(split, init)
            uncal_test_result, _ = main(split, init, eval_type_list, parent_dir, args)
            for eval_type, eval_metric in uncal_test_result.items():
                for metric in eval_metric:
                    uncal_test_total[eval_type][metric].extend(uncal_test_result[eval_type][metric])
                    

    # print results
    for name, result in zip(['Uncal'], [uncal_test_total]):
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
    print('---------------------------------------------------------------------')