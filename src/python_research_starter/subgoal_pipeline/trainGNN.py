import torch
from torch_geometric.loader import DataLoader
from python_research_starter.subgoal_pipeline.GraphImportanceGNN import GraphImportanceGNN
from python_research_starter.subgoal_pipeline.GraphPairDataset import GraphPairDataset

# Load data
dataset = GraphPairDataset("dataset.pkl")

# Constants
x1_count = (dataset[0].x[:, -1] == 0).sum().item()
N_GRAPH1 = x1_count

# Model and datapipeline initialization
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = GraphImportanceGNN(in_channels=dataset[0].x.size(1), hidden_channels=64)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Train loop
epochs = 1000
#model.train()
for epoch in range(epochs):
    total_loss = 0
    for data in loader:
        print()
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
    
    if epoch % 10 == 0:
        # More comprehensive save
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }
        torch.save(checkpoint, "graph_importance_checkpoint.pt")

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")