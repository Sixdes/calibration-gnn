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
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from src.model.model import create_model
from src.utils import set_global_seeds, arg_parse, name_model, metric_mean, metric_std
from src.calibloss import NodewiseECE, NodewiseBrier, NodewiseNLL
from src.data.data_utils import load_data
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

def main(split, init, gamma_best, args):
    # Evaluation
    val_result = defaultdict(list)
    test_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        epochs = 2000
        lr = 0.01 #0.05
        model_name = name_model(fold, args)
        
        # Early stopping
        patience = 100
        vlss_mn = float('Inf')
        vacc_mx = 0.0
        state_dict_early_model = None
        curr_step = 0
        best_result = {}

        dataset = load_data(args.dataset, args.split_type, split, fold)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = dataset.data.to(device)

        model = create_model(dataset, args).to(device)
        
        # print(model)
        # criterion = torch.nn.CrossEntropyLoss()
        gcl_loss_func = GCL_Loss(device, gamma=gamma_best)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wdecay)

        for i in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            # gcl_loss
            loss_gcl = gcl_loss_func(logits[data.train_mask], data.y[data.train_mask]) 
            loss_gcl.backward()
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
                    eval = NodewiseMetrics(logits, data.y, mask)
                    acc, nll, brier, ece = eval.acc(), eval.nll(), eval.brier(), eval.ece()
                    accs.append(acc); nlls.append(nll); briers.append(brier); eces.append(ece)                    

                ### Early stopping
                val_acc = acc; val_loss = nll
                if val_acc >= vacc_mx or val_loss <= vlss_mn:
                    if val_acc >= vacc_mx and val_loss <= vlss_mn:
                        state_dict_early_model = copy.deepcopy(model.state_dict())
                        b_epoch = i
                        best_result.update({'log_prob':logits,
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

        dir = Path(os.path.join('model_gcl-grid-1117', args.dataset, args.split_type, 'split'+str(split), 
                                'init'+ str(init)))
        dir.mkdir(parents=True, exist_ok=True)
        file_name = dir / (model_name + '.pt')
        # print(f"Model_name: {model_name}, Best gamma after training: {gamma_best}, best val accuracy: {vacc_mx}")
        torch.save(state_dict_early_model, file_name)
    return val_result, test_result

class GCL_Loss(nn.Module):
    def __init__(self, device, gamma):
        super(GCL_Loss, self).__init__()
        self.gamma = gamma
        self.device = device
        # gamma = torch.tensor(gamma).to(device)
        # gamma.requires_grad = True
        # self.gamma = nn.Parameter(gamma)

    def forward(self, logits, true_labels):

        probs = F.softmax(logits, dim=1)
        true_probs = probs.gather(1, true_labels.view(-1, 1)).view(-1)

        nll_loss = -true_probs.log()
        loss = (1 + self.gamma * true_probs) * nll_loss  

        return loss.mean()


# Grid search: gamma
def grid_search(model, data, gamma_values, args):
    # gamma_values=[0.0003, 0.001, 0.005, 0.01]
    best_gamma = None
    best_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for gamma in gamma_values:
        print(f"Training with gamma: {gamma}")
        criterion = GCL_Loss(device, gamma=gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.wdecay)

        for epoch in range(300):  # train 300 epoch
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_loss = criterion(logits[data.val_mask], data.y[data.val_mask])

        print(f"Validation Loss: {val_loss.item()}")

        # 记录最佳 gamma
        if val_loss < best_loss:
            best_loss = val_loss
            best_gamma = gamma

    return best_gamma, best_loss

if __name__ == '__main__':
    args = arg_parse()
    set_global_seeds(args.seed)
    max_splits,  max_init = 5, 5
    # grid search gamma
    gamma_values = [0.0003, 0.001, 0.005, 0.01, 0.015, 0.2]
    dataset_grid = load_data(args.dataset, args.split_type, 0, 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_grid = dataset_grid.data.to(device)

    model = create_model(dataset_grid, args).to(device)
    gamma_best, best_loss = grid_search(model, data_grid, gamma_values, args)
    print(f"the best gamma of {args.dataset} with {args.model} is {gamma_best}, the val loss is {best_loss}")

    val_total_result = {'acc':[], 'nll':[]}
    test_total_result = {'acc':[], 'nll':[]}
    for split in range(max_splits):
        for init in range(max_init):
            val_result, test_result = main(split, init, gamma_best, args)
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

