"""
Xem có bao nhiêu file chung.
"""
# clean_file = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\database\database_clean.tsv"
# database8_file = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\database\database8.tsv"
# output_file = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\database\only_in_db8.tsv"

# def get_filenames_and_lines(tsv_path):
#     with open(tsv_path, "r", encoding="utf-8") as f:
#         lines = [line for line in f if line.strip()]
#         filenames = set(line.strip().split('\t')[0] for line in lines)
#     return filenames, lines

# files_db, _ = get_filenames_and_lines(clean_file)
# files_db8, lines_db8 = get_filenames_and_lines(database8_file)

# # 1. File có trong database8.tsv mà không có trong database_clean.tsv
# only_in_db8 = files_db8 - files_db

# # Lọc các dòng tương ứng và ghi ra file mới
# with open(output_file, "w", encoding="utf-8") as fout:
#     for line in lines_db8:
#         filename = line.strip().split('\t')[0]
#         if filename in only_in_db8:
#             fout.write(line)

# print(f"Đã ghi {len(only_in_db8)} dòng vào {output_file}")

import pandas as pd

input_file = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\database\only_in_db8_filteredV3.tsv"
output_file = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\database\only_in_db8_filteredV4.tsv"


# Lọc duration >= 1
df = pd.read_csv(input_file, sep='\t', header=None, names=['filename', 'duration', 'text'], on_bad_lines='skip')
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
df = df[df['duration'] >= 1]
# Lọc text có từ 4 đến 30 từ
df = df[df['text'].str.split().apply(len).between(4, 30)]

# Lưu lại file đã lọc
df.to_csv(output_file, sep='\t', index=False, header=False)

print(f"Đã ghi {len(df)} dòng vào {output_file}")