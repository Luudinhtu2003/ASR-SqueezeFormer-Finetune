import pandas as pd

# Đường dẫn file
tsv_file = "fpt_clean_english.tsv"
remove_text_file = "high_wer_texts.txt"
output_file = "fpt_vn_filter_wer.tsv"

# Đọc file TSV
df = pd.read_csv(tsv_file, sep="\t")

# Đọc các dòng text cần loại bỏ
with open(remove_text_file, "r", encoding="utf-8") as f:
    remove_texts = set(line.strip() for line in f if line.strip())

# Lọc các dòng không có trong danh sách cần loại bỏ
filtered_df = df[~df['text'].isin(remove_texts)]

# Lưu lại file đã lọc
filtered_df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")
