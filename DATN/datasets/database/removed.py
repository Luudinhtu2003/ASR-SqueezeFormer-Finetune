"""
Loại bỏ những file WER cao.
"""
import pandas as pd

# Đường dẫn file
tsv_file = "only_in_db8_filteredV4.tsv"
remove_text_file = "high_wer_conLai.txt"
output_file = "only_in_db8_filteredV3.tsv"

# Đọc file TSV
df = pd.read_csv(tsv_file, sep="\t")

# Đọc các dòng text cần loại bỏ
with open(remove_text_file, "r", encoding="utf-8") as f:
    remove_texts = set(line.strip() for line in f if line.strip())

# Lọc các dòng không có trong danh sách cần loại bỏ
filtered_df = df[~df['text'].isin(remove_texts)]

# Lưu lại file đã lọc
filtered_df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")

# import csv
# import os

# input_file = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivos\valid.tsv"
# temp_file = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivos\validV2.tsv"

# with open(input_file, 'r', encoding='utf-8') as fin, \
#      open(temp_file, 'w', encoding='utf-8', newline='') as fout:
#     reader = csv.reader(fin, delimiter='\t')
#     writer = csv.writer(fout, delimiter='\t')
#     for row in reader:
#         text = row[2]  # Cột thứ 3 là text (theo ví dụ file của bạn)
#         if len(text.strip().split()) >= 2:
#             writer.writerow(row)

# # Ghi đè file gốc bằng file tạm
# os.replace(temp_file, input_file)