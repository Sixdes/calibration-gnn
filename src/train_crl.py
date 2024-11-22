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
from src.data.data_utils import load_data
from tqdm import tqdm
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

def main(split, init, args):
    # Evaluation
    val_result = defaultdict(list)
    test_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))
    alpha = args.alpha
    lmbda = args.lmbda

    for fold in range(max_fold):
        epochs = 2000
        # lr = 0.01 #0.05
        lr = 0.005
        model_name = name_model(fold, args)
        
        # Early stopping
        patience = 100
        vlss_mn = float('Inf')
        vacc_mx = 0.0
        state_dict_early_model = None
        curr_step = 0
        best_result = {}

        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(dataset, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wdecay)
        
        # print(model)
        data = data.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss_nll = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss_cal = cal_loss(
                data.y[data.train_mask], logits[data.train_mask], lmbda, i, epochs, device
            )
            
            loss = alpha * loss_nll + (1.0 - alpha) * loss_cal 
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
        dir = Path(os.path.join('model_crl', args.dataset, args.split_type, 'split'+str(split), 
                                'init'+ str(init)))
        dir.mkdir(parents=True, exist_ok=True)
        file_name = dir / (model_name + '.pt')
        torch.save(state_dict_early_model, file_name)
    return val_result, test_result

def cal_loss(y_true, logits, lmbda, epoch, epochs, device):
    def calculate_confidence_vec(confidence, y_pred, y_true, device, bin_num=15):
        def compute_binned_acc_conf(
            conf_thresh_lower, conf_thresh_upper, conf, pred, true, device
        ):
            filtered_tuples = [
                x
                for x in zip(pred, true, conf)
                if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper
            ]
            if len(filtered_tuples) < 1:
                return (
                    torch.tensor(0.0).to(device),
                    torch.tensor(0.0, requires_grad=True).to(device),
                    torch.tensor(0).to(device),
                )
            else:
                correct = len(
                    [x for x in filtered_tuples if x[0] == x[1]]
                )  # How many correct labels
                len_bin = torch.tensor(len(filtered_tuples)).to(
                    device
                )  # How many elements fall into the given bin
                avg_conf = (
                    torch.sum(torch.stack([x[2] for x in filtered_tuples])) / len_bin
                )  # Avg confidence of bin
                accuracy = (torch.tensor(correct, dtype=torch.float32) / len_bin).to(
                    device
                )  # Accuracy of bin
            return accuracy, avg_conf, len_bin

        bin_size = torch.tensor(1.0 / bin_num)
        upper_bounds = torch.arange(bin_size, 1 + bin_size, bin_size)

        accuracies = []
        num_in_each_bin = []

        for conf_thresh in upper_bounds:
            acc, avg_conf, len_bin = compute_binned_acc_conf(
                conf_thresh - bin_size, conf_thresh, confidence, y_pred, y_true, device
            )
            accuracies.append(acc)
            num_in_each_bin.append(len_bin)

        acc_all = []
        for conf in confidence:
            # idx = int(conf // (1 / bin_num))
            idx = int(torch.div(conf, (1 / bin_num)))
            acc_all.append(accuracies[idx])

        return torch.stack(acc_all), torch.stack(num_in_each_bin)

    def calculate_cal_term(acc_vector, conf_vector, num_in_each_bin):
        bin_error = acc_vector * torch.log(conf_vector)
        cal_term = -torch.sum(bin_error)
        return cal_term

    probs = F.softmax(logits, dim=1)
    y_pred = torch.max(logits, axis=1)[1]
    confidence = torch.max(probs, axis=1)[0]
    acc_vector, num_in_each_bin = calculate_confidence_vec(
        confidence, y_pred, y_true, device
    )
    cal_term = calculate_cal_term(acc_vector, confidence, num_in_each_bin)

    lmbda = torch.tensor(lmbda)
    annealing_coef = torch.min(lmbda, torch.tensor(lmbda * (epoch + 1) / epochs))

    return cal_term * annealing_coef


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    max_splits,  max_init = 5, 5


    val_total_result = {'acc':[], 'nll':[]}
    test_total_result = {'acc':[], 'nll':[]}
    for split in range(max_splits):
        for init in range(max_init):
            val_result, test_result = main(split, init, args)
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

