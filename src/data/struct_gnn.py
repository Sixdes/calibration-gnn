import torch
import torch_geometric
from torch_geometric.data import Data
import random

def modify_del_graph(data, delete_ratio=0.1):
    """
    Modify the graph by randomly deleting and adding edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        delete_ratio (float): The ratio of edges to delete based on label inconsistency.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    dtype = edge_index.dtype
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    
    # Get node labels
    labels = data.y.cpu().numpy()
    
    # Randomly delete edges with inconsistent labels
    edges_to_delete = []
    for u, v in edges:
        if labels[u] != labels[v] and random.random() < delete_ratio:
            edges_to_delete.append([u, v])
            edges_to_delete.append([v, u])
            edges.remove([u, v])
            edges.remove([v, u])
    
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    # Create a new Data object with the modified edge_index
    # modified_data = Data(x=data.x, edge_index=new_edge_index, y=data.y).to(device)
    
    return new_edge_index

def modify_add_graph(data, add_iteration=0):
    """
    Modify the graph by randomly deleting and adding edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        add_ratio (float): The ratio of edges to add between nodes with the same label.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    dtype = edge_index.dtype
    
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    labels = data.y.cpu().numpy()

    # Randomly add edges between nodes with the same label
    num_nodes = data.num_nodes
    edges_to_add = []
    idx = 0
    for _ in range(add_iteration):  
        u, v = random.sample(range(num_nodes), 2)
        if labels[u] == labels[v] and [u, v] not in edges and [v, u] not in edges:
            edges_to_add.append([u, v])
            edges_to_add.append([v, u])
            edges.append([u, v])
            edges.append([v, u])
            idx = idx + 1
    print(f'add the num of edges: {idx}')
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    return new_edge_index


def modify_del_valid_graph(data, delete_ratio=0.1):
    """
    Modify the graph by randomly deleting train-val edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        delete_ratio (float): The ratio of edges to delete based on label inconsistency.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    dtype = edge_index.dtype
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    labels = data.y.cpu().numpy()

    valid_node_indices = torch.nonzero(data.train_mask | data.val_mask).view(-1)
    
    # Randomly delete edges with inconsistent labels
    edges_to_delete = []
    for u, v in edges:
        if (u in valid_node_indices) and (v in valid_node_indices):
            if labels[u] != labels[v] and random.random() < delete_ratio:
                edges_to_delete.append([u, v])
                edges_to_delete.append([v, u])
                edges.remove([u, v])
                edges.remove([v, u])
    
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    return new_edge_index


def modify_add_valid_graph(data, add_iteration=0):
    """
    Modify the graph by randomly adding train-val edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        add_ratio (float): The ratio of edges to add between nodes with the same label.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    dtype = edge_index.dtype
    
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    labels = data.y.cpu().numpy()
    valid_node_indices = torch.nonzero(data.train_mask | data.val_mask).view(-1)

    # Randomly add edges between nodes with the same label
    num_nodes = data.num_nodes
    edges_to_add = []
    idx = 0
    for _ in range(add_iteration):  
        u, v = random.sample(valid_node_indices.tolist(), 2)
        if labels[u] == labels[v] and [u, v] not in edges and [v, u] not in edges:
            edges_to_add.append([u, v])
            edges_to_add.append([v, u])
            edges.append([u, v])
            edges.append([v, u])
            idx = idx + 1

    print(f'add the num of edges: {idx}')
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    return new_edge_index

def modify_add_valid_graph_ratio(data, add_ratio=0.1):
    """
    Modify the graph by randomly adding train-val edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        add_ratio (float): The ratio of edges to add between nodes with the same label.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    dtype = edge_index.dtype
    
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    labels = data.y.cpu().numpy()
    valid_node_indices = torch.nonzero(data.train_mask | data.val_mask).view(-1)

    # Randomly add edges between nodes with the same label
    num_nodes = data.num_nodes
    edges_to_add = []
    valid_edges_count = 0
    for u, v in edge_index.t().tolist():
        if u in valid_node_indices and v in valid_node_indices:
            valid_edges_count += 1

    num_edges_to_add_1 = int(valid_edges_count * add_ratio / 2)
    num_edges_to_add_2 = int(len(valid_node_indices) * (len(valid_node_indices) - 1) * add_ratio / 2)

    
    for _ in range(num_edges_to_add_1):  
        u, v = random.sample(valid_node_indices.tolist(), 2)
        if labels[u] == labels[v] and [u, v] not in edges and [v, u] not in edges:
            edges_to_add.append([u, v])
            edges_to_add.append([v, u])
            edges.append([u, v])
            edges.append([v, u])
    
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    return new_edge_index

def modify_add_graph_ratio(data, add_ratio=0.1):
    """
    Modify the graph by randomly deleting and adding edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        add_ratio (float): The ratio of edges to add between nodes with the same label.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    dtype = edge_index.dtype
    
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    labels = data.y.cpu().numpy()

    # Randomly add edges between nodes with the same label
    num_nodes = data.num_nodes
    edges_to_add = []

    # num_edges_to_add = int(num_nodes * (num_nodes - 1) * add_ratio
    num_edges_to_add = int(num_edges * add_ratio / 2)
    
    for _ in range(num_edges_to_add):  
        u, v = random.sample(range(num_nodes), 2)
        if labels[u] == labels[v] and [u, v] not in edges and [v, u] not in edges:
            edges_to_add.append([u, v])
            edges_to_add.append([v, u])
            edges.append([u, v])
            edges.append([v, u])
    
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    # Create a new Data object with the modified edge_index
    # modified_data = Data(x=data.x, edge_index=new_edge_index, y=data.y).to(device)
    
    return new_edge_index


if __name__ == '__main__':
    # Example usage
    num_nodes = 10
    num_edges = 20
    num_classes = 3

    # Create a random graph for demonstration
    x = torch.randn(num_nodes, 5)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edges
    y = torch.randint(0, num_classes, (num_nodes,))  # Random labels

    data = Data(x=x, edge_index=edge_index, y=y)

    # Modify the graph
    delete_graph_data = modify_del_graph(data, delete_ratio=0.6)
    add_graph_data = modify_add_graph(data, add_ratio=0.2)

    print("Original edges:")
    print(data.edge_index)
    print("Modified delete edges:")
    print(delete_graph_data.edge_index)
    print("Modified add edges:")
    print(add_graph_data.edge_index)