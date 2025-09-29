import pandas as pd

# Đọc file .tsv (giả sử tên file là 'data.tsv')
df = pd.read_csv("fpt_clean_english.tsv", sep="\t")

# Lấy cột 'text'
texts = df['text'].tolist()

with open("texts_fpt.txt", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")
