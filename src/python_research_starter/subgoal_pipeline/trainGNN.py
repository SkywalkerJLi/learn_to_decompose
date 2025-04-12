import torch
from torch_geometric.loader import DataLoader
from python_research_starter.subgoal_pipeline.GNN_Models import get_model
from python_research_starter.subgoal_pipeline.GraphPairDataset import GraphPairDataset
from sklearn.metrics import f1_score

# Load data
dataset = GraphPairDataset("dataset.pkl")
print(len(dataset))

# Constants
x1_count = (dataset[0].x[:, -1] == 0).sum().item()
N_GRAPH1 = x1_count

# Model and datapipeline initialization
loader = DataLoader(dataset, batch_size=16, shuffle=True)

in_channels = dataset[0].x.size(1)
hidden_channels = 32
model = get_model("gat", in_channels, hidden_channels, heads=4)

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Train loop
epochs = 100
model.train()
for epoch in range(epochs):
    total_loss = 0
    all_preds = []
    all_targets = []
    for data in loader:
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)  # shape: [total_nodes]

        batch_size = data.num_graphs
        total_graph1_nodes = N_GRAPH1 * batch_size
        graph1_mask = (data.x[:, -1] == 0)

        out_graph1 = out[graph1_mask]
        y_graph1 = data.y

        loss = loss_fn(out_graph1, y_graph1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Store predictions for F1 score (threshold at 0)
        preds = (out_graph1 > 0).long().cpu().numpy()  # BCEWithLogitsLoss expects raw logits
        targets = y_graph1.long().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    f1 = f1_score(all_targets, all_preds)
    
    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }
        torch.save(checkpoint, "gat_graph_importance_checkpoint.pt")

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}, F1 = {f1:.4f}")