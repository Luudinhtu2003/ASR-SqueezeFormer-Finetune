texts = []
with open('Ftest_vivos.tsv', 'r', encoding='utf-8') as f:
    header = next(f)  # bỏ dòng header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            texts.append(parts[2])
        else:
            print(f"Lỗi dòng: {line.strip()}")  # in ra dòng lỗi để kiểm tra

# Ghi vào file txt
with open('text_only_test.txt', 'w', encoding='utf-8') as f_out:
    for text in texts:
        f_out.write(text + '\n')
