import torch
import torch.nn.functional as F
from src.data.data_utils import load_data, load_node_to_nearest_training

import os
import copy
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch_geometric.nn import GCNConv, GATConv
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Dataset
from torch_geometric.nn import MLP
from gat_conv_weight import GATConv_weight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, attention_head, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        attention_head = [1] + attention_head
        layer_list = []
        for i in range(len(self.feature_list)-1):
            concat = False if i == num_layers-1 else True 
            layer_list.append(["conv"+str(i+1), GATConv(self.feature_list[i]* attention_head[i], self.feature_list[i+1], 
                                                        heads=attention_head[i+1], dropout=drop_rate, concat=concat)])
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(len(self.feature_list)-1):
            x = F.dropout(x, self.drop_rate, self.training)
            x = self.layer_list["conv"+str(i+1)](x, edge_index)
            if i < len(self.feature_list)-2:
                x = F.elu(x)
        return x

class GAT_edgeweight(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, heads, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.conv1 = GATConv_weight(in_channels, num_hidden, heads, dropout=drop_rate)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv_weight(num_hidden * heads, num_classes, heads=1,
                             concat=False, dropout=drop_rate)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).cuda()

        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x_t = F.elu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        layer_list = []

        for i in range(len(self.feature_list)-1):
            layer_list.append(["conv"+str(i+1), GCNConv(self.feature_list[i], self.feature_list[i+1])])
        
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).cuda()
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        for i in range(len(self.feature_list)-1):
            x = self.layer_list["conv"+str(i+1)](x, edge_index, edge_weight)
            if i < len(self.feature_list)-2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x

class Edge_Weight(torch.nn.Module):
    def __init__(self, model, out_channels, dropout):
        super(Edge_Weight, self).__init__()
        self.model = model
        self.extractor = MLP([out_channels*2, out_channels*4, 1], dropout=dropout)

        for para in self.model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.get_weight(x, edge_index)
        print('edge_weight requires_grad: ', edge_weight.requires_grad)
        logist = self.model(x, edge_index, edge_weight)
        print('logist requires_grad: ', logist.requires_grad)
        return logist
    
    def fit(self, data, train_mask, test_mask, wdecay, lr=0.01, edge_weight=None, verbose=False):
        self.to(device)
        self.optimizer = torch.optim.Adam(self.extractor.parameters(),lr=lr, weight_decay=wdecay)
        fit_calibration_dcgc(self, data, train_mask, test_mask, edge_weight=edge_weight, verbose=verbose)
        return self

    def get_weight(self, x, edge_index):

        emb = self.model(x, edge_index)
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        f12 = torch.cat([f1, f2], dim=-1)
        edge_weight = self.extractor(f12)
        return edge_weight.relu()

def fit_calibration_dcgc(temp_model, data, train_mask, test_mask, edge_weight=None, patience = 100, verbose=False):
    """
    Train calibrator dcgc
    """    
    vlss_mn = float('Inf')

    labels = data.y
    model_dict = temp_model.state_dict()
    parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(2000):

        temp_model.train()
        temp_model.optimizer.zero_grad()
        logits = temp_model(data.x, data.edge_index, edge_weight)
        print()
        # Post-hoc calibration set the classifier to the evaluation mode
        # temp_model.model.eval()
        # assert not temp_model.model.training
        
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # print(logits.requires_grad)
        # print(loss)
        print('loss requires_grad: ', loss.requires_grad)
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            # logits = temp_model(data.x, data.edge_index, edge_weight)
            
            val_loss = F.cross_entropy(logits[test_mask], labels[test_mask])
            if val_loss <= vlss_mn:
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                # for debug
                preds = torch.argmax(logits, dim=1)[test_mask]
                acc = torch.mean((preds == data.y[test_mask]).to(torch.get_default_dtype())).item()
                ece = ECELoss(logits[test_mask], data.y[test_mask])
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
            if verbose:
                print(f'Epoch: : {epoch+1:03d}, Accuracy: {acc:.4f}, NNL: {val_loss:.4f}, ECE: {ece:.4f}')
    model_dict.update(state_dict_early_model)
    
    temp_model.load_state_dict(model_dict)

def ECELoss(logits, labels, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE).
    :param logits: Output logits from the model
    :param labels: True labels
    :param n_bins: Number of bins for ECE
    :return: float value of ECE
    """
    confidences = F.softmax(logits, dim=1).max(dim=1)[0]
    predictions = torch.argmax(logits, dim=1)
    errors = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = errors[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def load_base_data(name: str) -> Dataset:
    """
    name: str, the name of the dataset
    """
    transform = T.NormalizeFeatures()
    if name in ['Cora','Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/', name=name, transform=transform)
    return dataset

if __name__ == '__main__':
    file_name = 'model_labelrate/model/Cora/labelrate_20/GAT_run3.pt'
    checkpoint = torch.load(file_name)
    dataset = load_base_data('Cora')
    data = dataset.data.to(device)
    data.train_mask = checkpoint['train_mask']
    data.val_mask = checkpoint['val_mask']
    data.test_mask = checkpoint['test_mask']

    gcn = GCN(in_channels=dataset.num_features, num_classes=dataset.num_classes, num_hidden=64,
                    drop_rate=0.5, num_layers=2).to(device)
    gat = GAT_edgeweight(in_channels=dataset.num_features, num_classes = dataset.num_classes, num_hidden=8,
                    heads=8, drop_rate=0.5).to(device)
    gat1 = GAT(in_channels=dataset.num_features, num_classes = dataset.num_classes, num_hidden=8,
                    attention_head=[8,1], drop_rate=0.5, num_layers=2).to(device)
    
    # ew1 = Edge_Weight(gcn, dataset.num_classes, dropout=0.7).to(device)
    ew2 = Edge_Weight(gat, dataset.num_classes, dropout=0.7).to(device)
    
    gat.train()
    optimizer_gat = torch.optim.Adam(gat.parameters(), lr=0.005, weight_decay=5e-4)
    optimizer_gat.zero_grad()
    out = gat1(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer_gat.step()

    # ew1.fit(data, data.val_mask, data.train_mask, wdecay=0, edge_weight=None, verbose=False)
    ew2.fit(data, data.val_mask, data.train_mask, wdecay=0, edge_weight=None, verbose=False)