import torch
from sklearn.metrics import f1_score
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from python_research_starter.subgoal_pipeline.GNN_Models import get_model
from python_research_starter.subgoal_pipeline.GraphPairDataset import GraphPairDataset

# Load data
dataset_optimal = GraphPairDataset("val_dataset_optimal.pkl")
print(len(dataset_optimal))
print(dataset_optimal[0].x.size(1))
dataset = GraphPairDataset("dataset.pkl")
print(dataset[10].edge_attr.size(1))
# for i in range(10):
#     data = dataset_optimal.__getitem__(i)
#     print(data.x)
#     print(data.edge_index)
#     print(data.edge_attr)
