import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class EdgeMPNNLayer(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super().__init__(aggr='add')
        self.lin_node = torch.nn.Linear(node_in_dim, out_dim)
        self.lin_edge = torch.nn.Linear(edge_in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        # Project node features
        x = self.lin_node(x)

        # If needed, pad edge_attr to match self-loop edges
        if edge_attr.size(0) != edge_index.size(1):
            loop_attr = torch.zeros(x.size(0), device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        edge_attr = self.lin_edge(edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Combine source node features and edge features
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out  # or add residuals, norm, etc.

class EdgeMPNN(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.conv1 = EdgeMPNNLayer(node_in_dim, edge_in_dim, hidden_dim)
        self.conv2 = EdgeMPNNLayer(hidden_dim, edge_in_dim, hidden_dim)
        self.conv3 = EdgeMPNNLayer(hidden_dim, edge_in_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        return self.out(x).squeeze()

class GCNImportanceGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.fc1(x)
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
    elif model_type == "mpnn":
        edge_attr_channels = kwargs.get("edge_attr_channels", )
        return EdgeMPNN(in_channels, edge_attr_channels, hidden_channels)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
