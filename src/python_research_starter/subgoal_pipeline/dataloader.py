import pickle

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

demonstration_count = int(len(dataset) / 5)
print(demonstration_count)
# for graph, labels in dataset:
#     print(graph)
#     print(labels)