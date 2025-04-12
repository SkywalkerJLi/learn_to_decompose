import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCNImportanceGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.out(x)
        return x.squeeze()  # shape: [num_nodes]

class GATImportanceGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        x = F.elu(x)
        x = self.out(x)
        return x.squeeze()
    
def get_model(model_type: str, in_channels: int, hidden_channels: int, **kwargs):
    """
    Factory function to return a GNN model.
    
    Args:
        model_type (str): One of ["gcn", "gat"]
        in_channels (int): Input feature size
        hidden_channels (int): Hidden layer size
        kwargs: Extra keyword arguments (e.g., heads for GAT)
    
    Returns:
        torch.nn.Module: A GNN model
    """
    model_type = model_type.lower()
    if model_type == "gcn":
        return GCNImportanceGNN(in_channels, hidden_channels)
    elif model_type == "gat":
        heads = kwargs.get("heads", 4)
        return GATImportanceGNN(in_channels, hidden_channels, heads=heads)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
