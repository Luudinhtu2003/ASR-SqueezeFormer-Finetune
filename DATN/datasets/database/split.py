"""
Chia dữ liệu train, test, valid từ file TSV database, tỉ lệ là 6 2 2
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
df = pd.read_csv('clean_eng_wer.tsv', sep='\t')

# Shuffle dữ liệu
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Tính số lượng mẫu cho từng tập
total_len = len(df)
train_size = int(0.6 * total_len)
test_size = int(0.2 * total_len)
valid_size = total_len - train_size - test_size  # Đảm bảo đủ 100%

# Chia dữ liệu
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:train_size + test_size]
valid_df = df.iloc[train_size + test_size:]

# Lưu ra các file TSV
train_df.to_csv('clean_eng_wer_trainV2.tsv', sep='\t', index=False)
test_df.to_csv('clean_eng_wer_testV2.tsv', sep='\t', index=False)
valid_df.to_csv('clean_eng_wer_validV2.tsv', sep='\t', index=False)
