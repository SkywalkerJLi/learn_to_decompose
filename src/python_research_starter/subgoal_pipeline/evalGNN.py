import torch
from torch_geometric.loader import DataLoader
from python_research_starter.subgoal_pipeline.GNN_Models import get_model
from python_research_starter.subgoal_pipeline.GraphPairDataset import GraphPairDataset

# Load validation data
dataset = GraphPairDataset("val_dataset.pkl")
print(len(dataset))

# Constants
x1_count = (dataset[0].x[:, -1] == 0).sum().item()
N_GRAPH1 = x1_count

# Model and datapipeline initialization
loader = DataLoader(dataset, batch_size=8, shuffle=False)

in_channels = dataset[0].x.size(1)
hidden_channels = 32
model = get_model("gat", in_channels, hidden_channels, heads=4)


# Load the saved model weights
model.load_state_dict(torch.load("gat_graph_importance_checkpoint.pt")['model_state_dict'])
model.eval()  # Set model to evaluation mode

# Evaluation metrics
loss_fn = torch.nn.BCEWithLogitsLoss()
total_loss = 0
predictions = []
ground_truth = []

# No gradient calculation needed during evaluation
with torch.no_grad():
    for data in loader:
        # Forward pass
        out = model(data.x, data.edge_index, data.batch)
        
        # Get graph1 nodes using the indicator feature
        graph1_mask = (data.x[:, -1] == 0)
        out_graph1 = out[graph1_mask]
        y_graph1 = data.y
        
        # Calculate loss
        loss = loss_fn(out_graph1, y_graph1)
        total_loss += loss.item()
        
        # Store predictions and ground truth for metrics
        pred_probs = torch.sigmoid(out_graph1)  # Convert logits to probabilities
        predictions.append(pred_probs)
        ground_truth.append(y_graph1)

# Calculate average loss
avg_loss = total_loss / len(loader)
print(f"Test Loss: {avg_loss:.4f}")

# Convert lists to tensors
all_preds = torch.cat(predictions)
all_gt = torch.cat(ground_truth)

# Calculate additional metrics
binary_preds = (all_preds > 0.5).float()
accuracy = (binary_preds == all_gt).float().mean().item()
print(f"Accuracy: {accuracy:.4f}")

# Visualize node importance scores (optional)
for i in range(min(5, len(dataset))):  # Show first 5 examples
    data = dataset[i]
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        graph1_mask = (data.x[:, -1] == 0)
        importance_scores = torch.sigmoid(out[graph1_mask])
        
    print(f"\nExample {i+1}:")
    # print("Input graph: ", data.x)
    # print("Input edge_links", data.edge_index)
    print(f"Node importance scores: {importance_scores.numpy()}")
    print(f"One-hot scores: {importance_scores.numpy() > 0.5}" )
    print(f"True labels: {data.y.numpy()}")