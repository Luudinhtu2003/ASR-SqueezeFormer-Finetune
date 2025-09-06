"""
Dùng để kiểm tra việc chia train, test, valid có đúng không
Dành cho cả tập speaker và tập VIVOS
Dùng lại chỉ cần thay đổi đường dẫn tới file train, valid, test của speaker và Valid.
"""
#Kiểm tra xem có speaker nào trùng nhau giữa các file train, valid, test không
def extract_speakers(file_path):
    speakers = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts: continue
            filepath = parts[0]
            segments = filepath.split("\\")
            try:
                idx = segments.index("speaker")
                speaker_id = segments[idx + 1]
                speakers.add(speaker_id)
            except (ValueError, IndexError):
                continue
    return speakers

train_speakers = extract_speakers(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\speaker\Ftrain_speaker.tsv")
valid_speakers = extract_speakers(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\speaker\Fvalid_speaker.tsv")
test_speakers  = extract_speakers(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\speaker\Ftest_speaker.tsv")

# Kiểm tra giao nhau
print("🔁 Speaker trùng giữa train và valid:", train_speakers & valid_speakers)
print("🔁 Speaker trùng giữa train và test:", train_speakers & test_speakers)
print("🔁 Speaker trùng giữa valid và test:", valid_speakers & test_speakers)

# In số lượng
print("\n📊 Số lượng speaker:")
print("Train:", len(train_speakers))
print("Valid:", len(valid_speakers))
print("Test:", len(test_speakers))

#Đếm số lượng speaker trong VIVOS
with open(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vivos\Ftrain_vivos.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()

speaker_ids = set()
for line in lines:
    filepath = line.strip().split('\t')[0]
    parts = filepath.split("\\")
    try:
        idx = parts.index("waves")
        speaker_id = parts[idx + 1]
        speaker_ids.add(speaker_id)
    except (ValueError, IndexError):
        continue

print("Tổng số speaker:", len(speaker_ids))

def extract_speakers(file_path):
    speakers = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                # Lấy phần tên speaker từ đường dẫn
                path_parts = parts[0].split('\\')
                for part in path_parts:
                    if part.startswith('VIVOSSPK'):
                        speakers.add(part)
                        break
    return speakers

file1_speakers = extract_speakers(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vivos\Ftrain_vivos_train.tsv")
file2_speakers = extract_speakers(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vivos\Ftrain_vivos_valid.tsv")

common_speakers = file1_speakers.intersection(file2_speakers)

print("Các speaker trùng nhau:", common_speakers)
print("Số lượng speaker trùng nhau:", len(common_speakers))


# Xem có file nào trùng nhau giữa 3 file .tsv không
def get_audio_paths(tsv_file):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        return set(line.strip().split('\t')[0] for line in f if line.strip())

# Đọc các file .tsv
paths1 = get_audio_paths(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivosV3\train.tsv")
paths2 = get_audio_paths(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivosV3\test.tsv")
paths3 = get_audio_paths(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivosV3\valid.tsv")

# Tìm file trùng nhau
dup12 = paths1 & paths2
dup13 = paths1 & paths3
dup23 = paths2 & paths3
dup_all = paths1 & paths2 & paths3

# In kết quả
print("Số file trùng giữa file1 và file2:", len(dup12))
print("Số file trùng giữa file1 và file3:", len(dup13))
print("Số file trùng giữa file2 và file3:", len(dup23))
print("Số file trùng giữa cả 3 file:", len(dup_all))

# Nếu muốn in tên file trùng:
print("Danh sách file trùng giữa 3 file:", dup_all)
