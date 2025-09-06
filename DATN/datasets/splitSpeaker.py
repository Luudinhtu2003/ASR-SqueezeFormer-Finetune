
"""
File này dùng để chia speaker thành các file train, valid, test từ file speaker.tsv.
tỉ lệ đang để là 600 train, 80 valid, 80 test.
"""
import os

# Tính tổng số lượng Speaker trong file speaker.tsv
# speaker_ids = set()
# with open("speaker.tsv", "r", encoding="utf-8") as f:  # thay tên file nếu khác
#     for line in f:
#         parts = line.strip().split('\t')
#         if not parts or len(parts) < 1:
#             continue
#         filepath = parts[0]
#         segments = filepath.split("\\")
#         try:
#             idx = segments.index("speaker")
#             speaker_id = segments[idx + 1]  # lấy ID ngay sau "speaker"
#             speaker_ids.add(speaker_id)
#         except (ValueError, IndexError):
#             print("Không tìm thấy speaker ID trong dòng:", line)

# print(f"Số lượng speaker khác nhau: {len(speaker_ids)}")
import os
from collections import defaultdict

# Xem số file của speaker >= 9 files
# speaker_file_count = defaultdict(int)

# with open("speaker.tsv", "r", encoding="utf-8") as f:
#     for line in f:
#         parts = line.strip().split('\t')
#         if not parts or len(parts) < 1:
#             continue
#         filepath = parts[0]
#         segments = filepath.split("\\")
#         try:
#             idx = segments.index("speaker")
#             speaker_id = segments[idx + 1]  # Lấy ID ngay sau "speaker"
#             speaker_file_count[speaker_id] += 1
#         except (ValueError, IndexError):
#             print("Không tìm thấy speaker ID trong dòng:", line)

# # Lọc các speaker có từ 10 file trở lên
# speakers_with_10_or_more_files = {k: v for k, v in speaker_file_count.items() if v < 9}

# # In kết quả
# print(f"Số lượng speaker có từ 10 file trở lên: {len(speakers_with_10_or_more_files)}")
# Nếu muốn in chi tiết từng speaker:
# for speaker_id, count in sorted(speakers_with_10_or_more_files.items(), key=lambda x: -x[1]):
#     print(f"Speaker {speaker_id}: {count} files")


#Tạo file train, valid, test từ file speaker.tsv
# import os
# import random
# from collections import defaultdict

# # Bước 1: Gom các dòng theo speaker
# speaker_lines = defaultdict(list)

# with open(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\speaker\speaker.tsv", "r", encoding="utf-8") as f:
#     for line in f:
#         parts = line.strip().split('\t')
#         if len(parts) < 1:
#             continue
#         filepath = parts[0]
#         segments = filepath.split("\\")
#         try:
#             idx = segments.index("speaker")
#             speaker_id = segments[idx + 1]
#             speaker_lines[speaker_id].append(line.strip())
#         except (ValueError, IndexError):
#             print("Không tìm thấy speaker ID trong dòng:", line)

# # Bước 2: Chọn ngẫu nhiên speaker
# all_speakers = list(speaker_lines.keys())
# random.shuffle(all_speakers)

# train_speakers = all_speakers[:600]
# valid_speakers = all_speakers[600:600+80]
# test_speakers = all_speakers[600+80:600+80+80]

# # Bước 3: Ghi file train, valid, test
# with open("Ftrain_speaker.tsv", "w", encoding="utf-8") as train_f:
#     for spk in train_speakers:
#         for line in speaker_lines[spk]:
#             train_f.write(line + '\n')

# with open("Fvalid_speaker.tsv", "w", encoding="utf-8") as valid_f:
#     for spk in valid_speakers:
#         for line in speaker_lines[spk]:
#             valid_f.write(line + '\n')

# with open("Ftest_speaker.tsv", "w", encoding="utf-8") as test_f:
#     for spk in test_speakers:
#         for line in speaker_lines[spk]:
#             test_f.write(line + '\n')

# # Bước 4: Thống kê
# test_file_count = sum(len(speaker_lines[spk]) for spk in test_speakers)
# train_file_count = sum(len(speaker_lines[spk]) for spk in train_speakers)
# valid_file_count = sum(len(speaker_lines[spk]) for spk in valid_speakers)
# print(f"Tổng số file trong Ftest_speaker.tsv (80 speaker): {test_file_count}")
# print(f"Tổng số file trong Frain_speaker.tsv (600 speaker): {train_file_count}")  #F là đã sửa transcripts
# print(f"Tổng số file trong Fvalid_speaker.tsv (80 speaker): {valid_file_count}")


#========================================
# Tạo file train, valid, test từ file train_vivos.tsv
"""
tỉ lệ chia là 40 speaker cho train, 6 cho valid, 0 cho test (test đã có 1 folder riêng).
"""
import random
from collections import defaultdict

# Bước 1: Gom các dòng theo speaker
speaker_lines = defaultdict(list)

with open(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\train_vivos\train_vivos.tsv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 1:
            continue
        filepath = parts[0]
        segments = filepath.split("\\")
        try:
            idx = segments.index("waves")  # Sửa 'speaker' thành 'waves' cho đúng cấu trúc VIVOS
            speaker_id = segments[idx + 1]  # Ví dụ: 'VIVOSSPK06'
            speaker_lines[speaker_id].append(line.strip())
        except (ValueError, IndexError):
            print("❌ Không tìm thấy speaker ID trong dòng:", line)

# Bước 2: Chọn ngẫu nhiên speaker
all_speakers = list(speaker_lines.keys())
random.shuffle(all_speakers)

train_speakers = all_speakers[:40]
valid_speakers = all_speakers[40:46]
test_speakers  = all_speakers[46:]

# Bước 3: Ghi file train, valid, test
with open("Ftrain_vivos_train.tsv", "w", encoding="utf-8") as train_f:
    for spk in train_speakers:
        for line in speaker_lines[spk]:
            train_f.write(line + '\n')

with open("Ftrain_vivos_valid.tsv", "w", encoding="utf-8") as valid_f:
    for spk in valid_speakers:
        for line in speaker_lines[spk]:
            valid_f.write(line + '\n')

with open("Ftrain_vivos_test.tsv", "w", encoding="utf-8") as test_f:
    for spk in test_speakers:
        for line in speaker_lines[spk]:
            test_f.write(line + '\n')

# Bước 4: Thống kê
train_file_count = sum(len(speaker_lines[spk]) for spk in train_speakers)
valid_file_count = sum(len(speaker_lines[spk]) for spk in valid_speakers)
test_file_count  = sum(len(speaker_lines[spk]) for spk in test_speakers)

print(f"✅ Tổng số file trong Ftrain_vivos_train.tsv (40 speaker): {train_file_count}")
print(f"✅ Tổng số file trong Ftrain_vivos_valid.tsv (6 speaker): {valid_file_count}")
print(f"✅ Tổng số file trong Ftrain_vivos_test.tsv ({len(test_speakers)} speaker): {test_file_count}")
