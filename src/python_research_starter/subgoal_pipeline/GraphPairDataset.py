import pickle
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

class GraphPairDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, "rb") as f:
            self.raw_data = pickle.load(f)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        graph, labels, subgoal = self.raw_data[idx]

        state_data = torch.tensor(graph.nodes[1:], dtype=torch.float)
        subgoal_data = torch.tensor(subgoal.nodes[1:], dtype=torch.float)
        x = torch.cat([state_data, subgoal_data], dim = 0)

        # Add graph indicator as node feature
        indicator = torch.cat([
            torch.zeros(state_data.size(0), 1),  # 0 for graph1
            torch.ones(subgoal_data.size(0), 1)    # 1 for graph2
        ], dim=0)

        x = torch.cat([x, indicator], dim=1)

        edge_index_state = torch.tensor(graph.edge_links, dtype=torch.long).t().contiguous()
        edge_index_subgoal = torch.tensor(subgoal.edge_links, dtype=torch.long).t().contiguous()
        edge_index_subgoal += state_data.size(0)  # shift node indices for second graph

        edge_index = torch.cat([edge_index_state, edge_index_subgoal], dim=1)

        # Optional edge features
        if graph.edges is not None and subgoal.edges is not None:
            edge_attr_state = torch.tensor(graph.edges, dtype=torch.float)
            edge_attr_subgoal = torch.tensor(subgoal.edges, dtype=torch.float)
            edge_attr = torch.cat([edge_attr_state, edge_attr_subgoal], dim=0)
        else:
            edge_attr = None

        y = torch.tensor(labels, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        if edge_attr is not None:
            data.edge_attr = edge_attr

        return data