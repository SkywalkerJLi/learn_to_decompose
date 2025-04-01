from spmf import Spmf
from sympy.utilities.iterables import multiset_partitions


def create_edge_dict(num_nodes):
    edge_dict = {}
    index_dict = {}
    count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_dict[tuple((i, j))] = count
            index_dict[count] = tuple((i, j))
            count += 1
            edge_dict[tuple((j, i))] = count
            index_dict[count] = tuple((i, j))
            count += 1
    return edge_dict, index_dict

edge_links_to_index, index_to_edge_links  = create_edge_dict(6)

elements = [0, 1, 2, 3, 4, 5]
# Generate all partitions and assign them fixed numbers
all_partitions = list(multiset_partitions(elements, m=None))  # k=None means any number of groups

# Assign a unique index to each partition
index_to_partition = {i: part for i, part in enumerate(all_partitions)}

input_example_list = [
    [[1], [1, 2, 3], [1, 3], [4], [3, 6]],
    [[1, 4], [3], [2, 3], [1, 5]],
    [[5, 6], [1, 2], [4, 6], [3], [2]],
    [[5], [7], [1, 6], [3], [2], [3]],
]

spmf = Spmf(
    "VMSP",
    #input_direct=input_example_list,
    input_filename="newtest.txt",
    spmf_bin_location_dir="/Users/skywalkerli/Desktop/Princeton_2024_2025/Research/learn-to-decompose/src/",
    output_filename="output.txt",
    arguments=[0.6],
)
spmf.run()
patterns = spmf.to_pandas_dataframe()
for _, row in patterns.iterrows(): 
    pattern = row['pattern']
#     for edge_link_scene in pattern:
#         edge_link_list = edge_link_scene.split()
#         for edge_link in edge_link_list:
#             print(index_to_edge_links[int(edge_link)])
#         print('next scene')
#     print('new pattern')
    for partition in pattern:
        print(index_to_partition[int(partition)])
#print(patterns)
