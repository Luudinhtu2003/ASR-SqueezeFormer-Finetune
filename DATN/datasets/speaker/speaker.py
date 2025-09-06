def read_tsv(file_path):
    data = {}
    full_lines = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                path, duration, text = parts
                data[path] = text
                full_lines[path] = line.strip()
    return data, full_lines

# Đọc dữ liệu từ 2 file
original, original_lines = read_tsv('speaker.tsv')
modified, _ = read_tsv('Fspeaker.tsv')

# File text đã thay đổi
changed = []
# File bị xóa
removed = []

for path in original:
    if path in modified:
        if original[path] != modified[path]:
            changed.append(f"{path}\nOriginal: {original[path]}\nModified: {modified[path]}\n")
    else:
        removed.append(original_lines[path])  # lưu dòng đầy đủ

# Ghi ra file changed.txt
with open('changed.txt', 'w', encoding='utf-8') as f:
    f.write(f"Số lượng file đã chỉnh sửa: {len(changed)}\n\n")
    for entry in changed:
        f.write(entry + "\n")

# Ghi ra file removed.txt
with open('removed.txt', 'w', encoding='utf-8') as f:
    f.write(f"Số lượng file đã bị loại bỏ: {len(removed)}\n\n")
    for line in removed:
        f.write(line + "\n")
