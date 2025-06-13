import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class GraphPairDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, "rb") as f:
            self.raw_data = pickle.load(f)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        graph, labels, subgoal = self.raw_data[idx]

        state_data = torch.tensor(graph.nodes[1:], dtype=torch.float)

        # print(state_data)
        # print(graph.edge_links)
        # print(graph.edges)
        # print(subgoal)
        # print(labels)

        # Add indicator for each node in the state data if it is in the subgoal data
        # Also add indicator for which goal block is on top of the other for the edge attr
        for i, edge_link in enumerate(subgoal.edge_links):
            first_node, second_node = edge_link[0], edge_link[1]
            state_data[first_node][11] = 1
            state_data[second_node][11] = 1

        subgoal_edge_links = torch.zeros(
            (subgoal.edge_links.shape[0] * 2, subgoal.edge_links.shape[1])
        )
        edge_attr_subgoal = torch.zeros(
            (subgoal_edge_links.size(0), 4), dtype=torch.float
        )

        for i, edge_link in enumerate(subgoal.edge_links):
            subgoal_edge_links[i * 2] = torch.tensor(edge_link)
            reversed = edge_link[::-1].copy()
            subgoal_edge_links[i * 2 + 1] = torch.tensor(reversed)

            edge_attr_subgoal[i * 2][
                2
            ] = 1  # the first element in the subgoal index is above the second
            edge_attr_subgoal[i * 2 + 1][
                3
            ] = 1  # the second element in the subgoal index is below the first

        # If no edge links in scene graph, return just the subgoals
        if len(graph.edge_links) <= 0:
            # Pad both edge index and edge attr so input is the same for model
            edge_index = torch.nn.functional.pad(
                subgoal_edge_links, (0, 20 - subgoal_edge_links.size(1)), "constant"
            )
            edge_index = edge_index.long()

            # Add extra rows
            pad_rows = 20 - edge_attr_subgoal.size(0)
            new_rows = torch.zeros((pad_rows, edge_attr_subgoal.size(1)))
            edge_attr = torch.cat([edge_attr_subgoal, new_rows], dim=0)

            y = torch.tensor(labels, dtype=torch.float)

            data = Data(x=state_data, edge_index=edge_index, edge_attr=edge_attr, y=y)

            return data

        state_edge_links = torch.zeros(
            (graph.edge_links.shape[0] * 2, graph.edge_links.shape[1])
        )
        edge_attr_state = torch.zeros((state_edge_links.size(0), 4), dtype=torch.float)

        for i, edge_link in enumerate(graph.edge_links):
            state_edge_links[i * 2] = torch.tensor(edge_link)
            reversed = edge_link[::-1].copy()
            state_edge_links[i * 2 + 1] = torch.tensor(reversed)

            edge_attr_state[i * 2][
                0
            ] = 1  # the first element in the index is above the second
            edge_attr_state[i * 2 + 1][
                1
            ] = 1  # the second element in the index is below the first

        # print(state_edge_links)
        # print(edge_attr_state)
        # print(subgoal_edge_links)
        # print(edge_attr_subgoal)

        state_edge_links_rows = state_edge_links.view(state_edge_links.size(0), -1)
        subgoal_edge_links_rows = subgoal_edge_links.view(
            subgoal_edge_links.size(0), -1
        )

        # print(state_edge_links_rows)
        # print(subgoal_edge_links_rows)

        # Compare each edge in the subgoal against all rows in the scene graph
        mask = torch.any(
            (subgoal_edge_links_rows[:, None] == state_edge_links_rows).all(dim=2),
            dim=1,
        )

        # print(mask)
        # Extend the mask to match length of edge attr tensor
        extended_mask_state = torch.zeros(edge_attr_state.size(0), dtype=torch.bool)
        extended_mask_subgoal = torch.zeros(edge_attr_subgoal.size(0), dtype=torch.bool)

        # Fill in the values from the original tensor
        extended_mask_state[: min(edge_attr_state.size(0), len(mask))] = mask[
            : edge_attr_state.size(0)
        ]
        extended_mask_subgoal[: len(mask)] = mask

        # print(extended_mask_state)

        # print(edge_attr_state)
        # print(edge_attr_subgoal)

        # If an edge in the state is also in the subgoal, add the subgoal's attribute to the edge attribute
        edge_attr_state[extended_mask_state] = (
            edge_attr_state[extended_mask_state]
            + edge_attr_subgoal[extended_mask_subgoal]
        )
        # print(edge_attr_state)

        # Select edges and corresponding edge features in subgoal not in scene graph
        edge_index_diff = subgoal_edge_links_rows[~mask].t()
        edge_attr_diff = edge_attr_subgoal[~mask]

        edge_index_state = state_edge_links.t().contiguous()

        # Add additional edges if they are present in the subgoal but not in the scene graph
        edge_index = torch.cat([edge_index_state, edge_index_diff], dim=1)
        # Add additional goal edge features from the subgoal
        edge_attr = torch.cat([edge_attr_state, edge_attr_diff])

        # Pad both edge index and edge attr so input is the same for model
        edge_index = torch.nn.functional.pad(
            edge_index, (0, 20 - edge_index.size(1)), "constant"
        )
        edge_index = edge_index.long()

        # Add extra rows
        pad_rows = 20 - edge_attr.size(0)
        new_rows = torch.zeros((pad_rows, edge_attr.size(1)))
        edge_attr = torch.cat([edge_attr, new_rows], dim=0)

        # print("final data preprocess")
        # print(edge_index)
        # print(edge_attr)

        # assert True == False

        # print(state_data)
        # print(edge_index)
        # print(edge_attr)

        y = torch.tensor(labels, dtype=torch.float)

        data = Data(x=state_data, edge_index=edge_index, edge_attr=edge_attr, y=y)

        return data

    """
    Convert scene graphs and subgoal graphs into torch.geometric data for GNN prediction
    """

    def convert_data(scene_graph, goal_graph):
        graph, subgoal = scene_graph, goal_graph

        state_data = torch.tensor(graph.nodes[1:], dtype=torch.float)

        # Add indicator for each node in the state data if it is in the subgoal data
        # Also add indicator for which goal block is on top of the other for the edge attr
        for i, edge_link in enumerate(subgoal.edge_links):
            first_node, second_node = edge_link[0], edge_link[1]
            state_data[first_node][11] = 1
            state_data[second_node][11] = 1

        subgoal_edge_links = torch.zeros(
            (subgoal.edge_links.shape[0] * 2, subgoal.edge_links.shape[1])
        )
        edge_attr_subgoal = torch.zeros(
            (subgoal_edge_links.size(0), 4), dtype=torch.float
        )

        for i, edge_link in enumerate(subgoal.edge_links):
            subgoal_edge_links[i * 2] = torch.tensor(edge_link)
            reversed = edge_link[::-1].copy()
            subgoal_edge_links[i * 2 + 1] = torch.tensor(reversed)

            edge_attr_subgoal[i * 2][
                2
            ] = 1  # the first element in the subgoal index is above the second
            edge_attr_subgoal[i * 2 + 1][
                3
            ] = 1  # the second element in the subgoal index is below the first

        # If no edge links in scene graph, return just the subgoals
        if len(graph.edge_links) <= 0:
            # Pad both edge index and edge attr so input is the same for model
            edge_index = torch.nn.functional.pad(
                subgoal_edge_links, (0, 20 - subgoal_edge_links.size(1)), "constant"
            )
            edge_index = edge_index.long()

            # Add extra rows
            pad_rows = 20 - edge_attr_subgoal.size(0)
            new_rows = torch.zeros((pad_rows, edge_attr_subgoal.size(1)))
            edge_attr = torch.cat([edge_attr_subgoal, new_rows], dim=0)

            data = Data(x=state_data, edge_index=edge_index, edge_attr=edge_attr)

            return data

        state_edge_links = torch.zeros(
            (graph.edge_links.shape[0] * 2, graph.edge_links.shape[1])
        )
        edge_attr_state = torch.zeros((state_edge_links.size(0), 4), dtype=torch.float)

        for i, edge_link in enumerate(graph.edge_links):
            state_edge_links[i * 2] = torch.tensor(edge_link)
            reversed = edge_link[::-1].copy()
            state_edge_links[i * 2 + 1] = torch.tensor(reversed)

            edge_attr_state[i * 2][
                0
            ] = 1  # the first element in the index is above the second
            edge_attr_state[i * 2 + 1][
                1
            ] = 1  # the second element in the index is below the first

        state_edge_links_rows = state_edge_links.view(state_edge_links.size(0), -1)
        subgoal_edge_links_rows = subgoal_edge_links.view(
            subgoal_edge_links.size(0), -1
        )

        # Compare each edge in the subgoal against all rows in the scene graph
        mask = torch.any(
            (subgoal_edge_links_rows[:, None] == state_edge_links_rows).all(dim=2),
            dim=1,
        )

        # Extend the mask to match length of edge attr tensor
        extended_mask_state = torch.zeros(edge_attr_state.size(0), dtype=torch.bool)
        extended_mask_subgoal = torch.zeros(edge_attr_subgoal.size(0), dtype=torch.bool)

        # Fill in the values from the original tensor
        extended_mask_state[: min(edge_attr_state.size(0), len(mask))] = mask[
            : edge_attr_state.size(0)
        ]
        extended_mask_subgoal[: len(mask)] = mask

        # If an edge in the state is also in the subgoal, add the subgoal's attribute to the edge attribute
        edge_attr_state[extended_mask_state] = (
            edge_attr_state[extended_mask_state]
            + edge_attr_subgoal[extended_mask_subgoal]
        )
        # print(edge_attr_state)

        # Select edges and corresponding edge features in subgoal not in scene graph
        edge_index_diff = subgoal_edge_links_rows[~mask].t()
        edge_attr_diff = edge_attr_subgoal[~mask]

        edge_index_state = state_edge_links.t().contiguous()

        # Add additional edges if they are present in the subgoal but not in the scene graph
        edge_index = torch.cat([edge_index_state, edge_index_diff], dim=1)
        # Add additional goal edge features from the subgoal
        edge_attr = torch.cat([edge_attr_state, edge_attr_diff])

        # Pad both edge index and edge attr so input is the same for model
        edge_index = torch.nn.functional.pad(
            edge_index, (0, 20 - edge_index.size(1)), "constant"
        )
        edge_index = edge_index.long()

        # Add extra rows
        pad_rows = 20 - edge_attr.size(0)
        new_rows = torch.zeros((pad_rows, edge_attr.size(1)))
        edge_attr = torch.cat([edge_attr, new_rows], dim=0)

        data = Data(x=state_data, edge_index=edge_index, edge_attr=edge_attr)

        return data
