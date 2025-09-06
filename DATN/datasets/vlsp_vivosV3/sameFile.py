with open('train.tsv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

unique_lines = set()
duplicates = []

for line in lines:
    if line in unique_lines:
        duplicates.append(line)
    else:
        unique_lines.add(line)

if duplicates:
    print("Các dòng trùng lặp:")
    for dup in set(duplicates):
        print(dup.strip())
else:
    print("Không có dòng nào trùng lặp.")