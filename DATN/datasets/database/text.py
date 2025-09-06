"""
File này để lọc lấy các text từ tsv phục vụ cho language model.
"""

import pandas as pd

# Đường dẫn đến file TSV của bạn
tsv_path = 'database.tsv'  # Thay bằng đường dẫn thật

# Đọc file TSV (tab-separated)
df = pd.read_csv(tsv_path, sep='\t')

# Lấy cột 'text'
texts = df['text']

# Ghi từng dòng văn bản vào file txt
with open('text_database.txt', 'w', encoding='utf-8') as f:
    for line in texts:
        f.write(line.strip() + '\n')
