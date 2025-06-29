### Simple python parsing script that takes as input the output of the SPMF algorithm
### and outputs a list of list of the most common edge links as their index values

with open("output.txt", "r", encoding="utf-8") as file:
    for line in file:
        content = line[:-9]
        print(content)
        groups = [group.strip() for group in content.split("-1") if group.strip()]

        # Convert each group to a list of integers
        result = [list(map(int, group.split())) for group in groups]
        print(result)
