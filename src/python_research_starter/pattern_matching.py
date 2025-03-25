from spmf import Spmf

input_example_list = [
    [[1], [1, 2, 3], [1, 3], [4], [3, 6]],
    [[1, 4], [3], [2, 3], [1, 5]],
    [[5, 6], [1, 2], [4, 6], [3], [2]],
    [[5], [7], [1, 6], [3], [2], [3]],
]

spmf = Spmf(
    "VMSP",
    input_direct=input_example_list,
    spmf_bin_location_dir="/Users/skywalkerli/Desktop/Princeton_2024_2025/Research/learn-to-decompose/src/",
    output_filename="output.txt",
    arguments=[0.8],
)
spmf.run()
patterns = spmf.to_pandas_dataframe()
print(patterns)
