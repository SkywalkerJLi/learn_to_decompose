import torch
from torch_geometric.loader import DataLoader
from python_research_starter.subgoal_pipeline.GNN_Models import get_model
from python_research_starter.subgoal_pipeline.GraphPairDataset import GraphPairDataset
from sklearn.metrics import f1_score
from torch.utils.data import Subset

# Load data
dataset = GraphPairDataset("dataset.pkl")
dataset = GraphPairDataset("dataset_optimal.pkl")
print(len(dataset))

# Model and datapipeline initialization
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Small Sample Testing
# single_sample_dataset = Subset(dataset, [1, 2, 3, 4])  # Just the first five samples
# print(single_sample_dataset)
# single_loader = DataLoader(single_sample_dataset, batch_size=1, shuffle=False)

in_channels = dataset[0].x.size(1)
hidden_channels = 64
edge_attr_channels = dataset[0].edge_attr.size(1)

mp_model = get_model("mpnn", in_channels, hidden_channels, edge_attr_channels = edge_attr_channels)

# Data preprocessing
total_ones = 0
total_zeros = 0

for data in loader:
    y = data.y
    total_ones += (y == 1).sum().item()
    total_zeros += (y == 0).sum().item()

print(f"Total 1s: {total_ones}, Total 0s: {total_zeros}")
print(f"Positive ratio: {total_ones / (total_ones + total_zeros):.4f}")

pos_weight = torch.tensor([total_zeros / total_ones])
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mp_optimizer = torch.optim.Adam(mp_model.parameters(), lr=1e-3)


# Train loop
epochs = 200
mp_model.train()

# relevant_indicies = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 10])
# for data in single_loader:
    
#     # truncated_data = []
#     # for node in data.x:
#     #     truncated_data.append(node[relevant_indicies])
#     # truncated_data = torch.stack(truncated_data)
#     print(data.x)
#     # print(truncated_data)
#     print(data.edge_index)
#     print(data.edge_attr)
#     print(data.y)
#     print()
#     print()

for epoch in range(epochs):
    total_loss = 0
    all_preds = []
    all_targets = []
    for data in loader:
        # optimizer.zero_grad()
        mp_optimizer.zero_grad()

        out = mp_model(data.x, data.edge_index, data.edge_attr)

        # out = model(data.x, data.edge_index, data.batch)  # shape: [total_nodes]
        labels = data.y

        # print("model output: ", out)
        # print("ground truth: ", labels)

        loss = loss_fn(out, labels)
        loss.backward()
        # optimizer.step()
        mp_optimizer.step()
        total_loss += loss.item()

        # Store predictions for F1 score (threshold at 0.5)
        pred_probs = torch.sigmoid(out).detach()  # Convert logits to probabilities
        pred = (pred_probs > 0.5).float()
        targets = labels.long().cpu().numpy()

        if epoch == 990:
            print("prediction: ",  pred_probs)
            print("binary predictions: ", (pred_probs > 0.5).float())
            print("targets:",  targets)
        # print("ground truth : ", targets)

        all_preds.extend(pred)
        all_targets.extend(targets)

    f1 = f1_score(all_targets, all_preds)

    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': mp_model.state_dict(),
            'optimizer_state_dict': mp_optimizer.state_dict(),
            'loss': total_loss,
        }
        torch.save(checkpoint, "mp_graph_importance_checkpoint.pt")

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}, F1 = {f1:.4f}")