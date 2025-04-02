def remove_duplicate_words(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(output_file, "w", encoding="utf-8") as file:
        for line in lines:
            words = line.split()
            unique_words = list(dict.fromkeys(words))
            unique_words.remove("-1")
            file.write(" ".join(unique_words) + "\n")


# Example usage
remove_duplicate_words("temp.txt", "temp_nodup.txt")
print("Duplicate words removed and saved in output.txt")
