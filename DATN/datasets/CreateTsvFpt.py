"""
Tạo tsv từ file transcriptAll.txt của FPT đã cung cấp.
"""
# input_path = "transcriptAll.txt"       # Đặt tên file gốc của bạn
# output_path = "fpt.tsv"     # File TSV đầu ra

# with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         parts = line.strip().split("|")
#         if len(parts) != 3:
#             continue
#         audio_path, text, time_range = parts
#         try:
#             start, end = map(float, time_range.split("-"))
#             duration = round(end - start, 5)
#             outfile.write(f"{audio_path}\t{duration}\t{text}\n")
#         except ValueError:
#             print(f"Lỗi xử lý thời gian dòng: {line}")

import re

input_file = "fpt.tsv"
output_file = "fpt_clean.tsv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    header = infile.readline()
    outfile.write(header)  # Ghi lại header
    
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue

        audio_path, duration, text = parts

        # Bỏ dòng nếu text chứa chữ số
        if re.search(r'\d', text):
            continue

        # Viết thường, loại bỏ dấu câu
        text = text.lower()
        text = re.sub(r"[,.?]", "", text)

        # Chuẩn hóa khoảng trắng
        text = " ".join(text.strip().split())

        outfile.write(f"{audio_path}\t{duration}\t{text}\n")

