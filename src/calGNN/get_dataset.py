from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def get_planetoid_dataset(data_name, normalize_features=True):
    dataset = Planetoid(root="/root/GATS/data", name=data_name)
    if normalize_features:
        dataset.transform = T.NormalizeFeatures()
    return dataset


from pathlib import Path
import numpy as np
import os
from torch_geometric.io.planetoid import index_to_mask

def get_planetoid_dataset_gats(data_name, split_type, split, fold, normalize_features=True):
    dataset = Planetoid(root="/root/GATS/data", name=data_name)
    if normalize_features:
        dataset.transform = T.NormalizeFeatures()
    load_split_from_numpy_files(dataset, data_name, split_type, split, fold)
    return dataset

def load_split_from_numpy_files(dataset, name, split_type, split, fold):
    """
    load train/val/test from saved k-fold split files
    """
    raw_dir = Path(os.path.join('/root/GATS/data','split', str(name), split_type))
    assert raw_dir.is_dir(), "Split type does not exist."
    split_file = f'{name.lower()}_split_{split}.npz'
    masks = np.load(raw_dir / split_file, allow_pickle=True)
    val_indices = masks['k_fold_indices'][fold]
    train_indices = np.concatenate(np.delete(masks['k_fold_indices'], fold, axis=0))
    test_indices = masks['test_indices']
    dataset.data.train_mask = index_to_mask(train_indices, dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(val_indices, dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(test_indices, dataset.data.num_nodes)

if __name__ == '__main__':
    max_fold = 3
    split = 1
    dataset_names = ['Cora', 'Citeseer', 'Pubmed']
    for dataset_name in dataset_names:
        print(dataset_name)
        dataset1 = get_planetoid_dataset(dataset_name)
        print('ori train:', dataset1.data.train_mask.sum().item())
        print('ori val:', dataset1.data.val_mask.sum().item())
        print('ori test:', dataset1.data.test_mask.sum().item())
        
        dataset2 = get_planetoid_dataset_gats(dataset_name, '5_3f_85', split, fold=1)
        print('gats train:', dataset2.data.train_mask.sum().item())
        print('gats val:', dataset2.data.val_mask.sum().item())
        print('gats test:', dataset2.data.test_mask.sum().item())