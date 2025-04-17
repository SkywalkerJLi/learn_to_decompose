import torch
from torch_geometric.loader import DataLoader
from python_research_starter.subgoal_pipeline.GNN_Models import get_model
from python_research_starter.subgoal_pipeline.GraphPairDataset import GraphPairDataset
from sklearn.metrics import f1_score
from torch.utils.data import Subset

# Load data
dataset = GraphPairDataset("dataset.pkl")
for i in range(10):
    data = dataset.__getitem__(i)
    print(data.x)
    print(data.edge_index)
    print(data.edge_attr)