# Đọc dữ liệu từ hai file
with open('fpt_65_trainV2.tsv', 'r', encoding='utf-8') as f:
    db8_lines = set(f.readlines())

with open('train.tsv', 'r', encoding='utf-8') as f:
    dbV1_lines = set(f.readlines())

# Lấy những dòng chỉ có trong database8.tsv
only_in_db8 = db8_lines - dbV1_lines
import csv
# Ghi ra file mới
with open('only_in_fpt_65_trainV2.tsv', 'w', encoding='utf-8') as f:
    f.writelines(only_in_db8)
