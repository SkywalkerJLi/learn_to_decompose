with open("test.txt", "r", encoding="utf-8") as file:
    content = file.read()
print(content)
modified_text = content.replace("]", "-2 z")  # Replace ',' with a new line
modified_text = modified_text.replace("z", " \n")
modified_text = modified_text.replace("-2", " -2")
with open("newtest.txt", "w", encoding="utf-8") as file:
    file.write(modified_text)
