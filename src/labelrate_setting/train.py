import os
import math
import random
import abc
import gc
import copy
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch 
import torch.nn.functional as F
from torch import Tensor, LongTensor
from src.model.model import create_model
from src.utils import set_global_seeds, arg_parse, name_model, metric_mean, metric_std
from src.calibloss import NodewiseECE, NodewiseBrier, NodewiseNLL
from src.data.data_utils import load_data_labelrate, load_data_cagcn
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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

def main(run, args):
    # Evaluation
    val_result = defaultdict(list)
    test_result = defaultdict(list)

    epochs = 2000
    lr = 0.01 #0.05
    
    # Early stopping
    patience = 100
    vlss_mn = float('Inf')
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    best_result = {}

    dataset = load_data_labelrate(args.dataset, args.labelrate, args.num_val_nodes, args.num_test_nodes)
    # dataset = load_data_cagcn(args.dataset, args.labelrate)
    data = dataset.data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(dataset, args).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wdecay)
    optimizer = torch.optim.Adam([
                dict(params=model.layer_list.conv1.parameters(), weight_decay=5e-4),
                dict(params=model.layer_list.conv2.parameters(), weight_decay=0)
            ], lr=lr)
    # print(model)
    data = data.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask]) 
        loss.backward()
        optimizer.step()

        # Evaluation on traing and val set
        accs = []
        nlls = []
        briers = []
        eces = []
        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index)
            log_prob = F.log_softmax(logits, dim=1).detach()

            for mask in [data.train_mask, data.val_mask]:
                eval = NodewiseMetrics(log_prob, data.y, mask)
                acc, nll, brier, ece = eval.acc(), eval.nll(), eval.brier(), eval.ece()
                accs.append(acc); nlls.append(nll); briers.append(brier); eces.append(ece)                    

            ### Early stopping
            val_acc = acc; val_loss = nll
            if val_acc >= vacc_mx or val_loss <= vlss_mn:
                if val_acc >= vacc_mx and val_loss <= vlss_mn:
                    state_dict_early_model = copy.deepcopy(model.state_dict())
                    b_epoch = i
                    best_result.update({'log_prob':log_prob,
                                        'acc':accs[1],
                                        'nll':nlls[1],
                                        'bs':briers[1],
                                        'ece':eces[1]})
                vacc_mx = np.max((val_acc, vacc_mx)) 
                vlss_mn = np.min((val_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
            if args.verbose:
                print(f'Epoch: : {i+1:03d}, Accuracy: {accs[0]:.4f}, NNL: {nlls[0]:.4f}, Brier: {briers[0]:.4f}, ECE:{eces[0]:.4f}')
                print(' ' * 14 + f'Accuracy: {accs[1]:.4f}, NNL: {nlls[1]:.4f}, Brier: {briers[1]:.4f}, ECE:{eces[1]:.4f}')
    
    eval = NodewiseMetrics(best_result['log_prob'], data.y, data.test_mask)
    acc, nll, brier, ece = eval.acc(), eval.nll(), eval.brier(), eval.ece()
    test_result['acc'].append(acc); test_result['nll'].append(nll); test_result['bs'].append(brier)
    test_result['ece'].append(ece)

    del best_result['log_prob']
    for metric in best_result:
        val_result[metric].append(best_result[metric])

    # print("best epoch is:", b_epoch)
    dir = Path(os.path.join('model_labelrate/model_hd16_op_drop', args.dataset, 'labelrate_'+str(args.labelrate)))
    dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model + "_run" + str(run)
    file_name = dir / (model_name + '.pt')
    torch.save({
                'model_state_dict': state_dict_early_model,
                'train_mask': data.train_mask,
                'val_mask': data.val_mask,
                'test_mask': data.test_mask,
                }, file_name)
    return val_result, test_result


if __name__ == '__main__':
    args = arg_parse()
    seed = np.random.randint(0, 10000)
    set_global_seeds(seed)
    print(f'the seed is {seed}')
    num_runs = 10
    
    print(f'---------------------------training with {args.model} on {args.dataset} - labelrate{args.labelrate}---------------------------')
    val_total_result = {'acc':[], 'nll':[]}
    test_total_result = {'acc':[], 'nll':[]}

    for run in range(num_runs):
        
        val_result, test_result = main(run, args)
        for metric in val_total_result:
            val_total_result[metric].extend(val_result[metric])
            test_total_result[metric].extend(test_result[metric])

    val_mean = metric_mean(val_total_result)
    test_mean = metric_mean(test_total_result)
    test_std = metric_std(test_total_result)
    print(f"Val  Accuracy: &{val_mean['acc']:.2f} \t" + " " * 8 +\
            f"NLL: &{val_mean['nll']:.4f}")
    print(f"Test Accuracy: &{test_mean['acc']:.2f}\pm{test_std['acc']:.2f} \t" + \
            f"NLL: &{test_mean['nll']:.4f}\pm{test_std['nll']:.4f}")

