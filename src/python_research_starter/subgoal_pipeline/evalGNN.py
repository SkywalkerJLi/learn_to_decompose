import torch
from torch_geometric.loader import DataLoader
from python_research_starter.subgoal_pipeline.GNN_Models import get_model
from python_research_starter.subgoal_pipeline.GraphPairDataset import GraphPairDataset

# Load validation data
dataset = GraphPairDataset("src/python_research_starter/subgoal_pipeline/datasets/val_dataset_optimal.pkl")
print(len(dataset))

# Model and datapipeline initialization
loader = DataLoader(dataset, batch_size=16, shuffle=False)

in_channels = dataset[0].x.size(1)
hidden_channels = 64
edge_attr_channels = dataset[0].edge_attr.size(1)

mp_model = get_model("mpnn", in_channels, hidden_channels, edge_attr_channels = edge_attr_channels)

# Load the saved model weights
mp_model.load_state_dict(torch.load("/Users/skywalkerli/Desktop/Princeton_2024_2025/Research/learn-to-decompose/src/python_research_starter/subgoal_pipeline/saved_models/mp_graph_importance_checkpoint.pt")['model_state_dict'])
mp_model.eval()  # Set model to evaluation mode

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

# Evaluation metrics
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
total_loss = 0
predictions = []
ground_truth = []

# No gradient calculation needed during evaluation
with torch.no_grad():
    for data in loader:
        # Forward pass
        out = mp_model(data.x, data.edge_index, data.edge_attr)
        
        # Get graph1 nodes using the indicator feature
        labels = data.y
        
        # Calculate loss
        loss = loss_fn(out, labels)
        total_loss += loss.item()
        
        # Store predictions and ground truth for metrics
        pred_probs = torch.sigmoid(out)  # Convert logits to probabilities
        predictions.append(pred_probs)
        ground_truth.append(labels)

# Calculate average loss
avg_loss = total_loss / len(loader)
print(f"Test Loss: {avg_loss:.4f}")

# Convert lists to tensors
all_preds = torch.cat(predictions)
all_gt = torch.cat(ground_truth)

# Calculate additional metrics
binary_preds = (all_preds > 0.8).float()
accuracy = (binary_preds == all_gt).float().mean().item()
print(f"Accuracy: {accuracy:.4f}")

# Visualize node importance scores (optional)
for i in range(min(5, len(dataset))):  # Show first 5 examples
    data = dataset[i]
    mp_model.eval()
    with torch.no_grad():
        out = mp_model(data.x, data.edge_index, data.edge_attr)
        importance_scores = torch.sigmoid(out)
        
    print(f"\nExample {i+1}:")
    print("Input graph: ", data.x)
    print("Input edge_links", data.edge_index)
    print(f"Node importance scores: {importance_scores.numpy()}")
    print(f"One-hot scores: {importance_scores.numpy() > 0.5}" )
    print(f"True labels: {data.y.numpy()}")