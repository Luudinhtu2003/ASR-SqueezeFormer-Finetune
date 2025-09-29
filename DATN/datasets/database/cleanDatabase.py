"""
File này dùng để loại bỏ những từ là tiếng Anh trong tập Fpt vì quá nhiều
"""
# input_file = "fpt_clean2.tsv"
# output_file = "fpt_clean3.tsv"

# count_with_w = 0
# total_duration_with_w = 0.0

# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     header = infile.readline()
#     outfile.write(header)  # giữ lại header

#     for line in infile:
#         parts = line.strip().split('\t')
#         if len(parts) != 3:
#             continue

#         audio_path, duration, text = parts
#         if 'j' in text:
#             count_with_w += 1
#             try:
#                 total_duration_with_w += float(duration)
#             except ValueError:
#                 print(f"Lỗi chuyển đổi duration ở dòng: {line}")
#             continue  # bỏ dòng này

#         outfile.write(line)

# print(f"Số dòng chứa 'z': {count_with_w}")
# print(f"Tổng duration các dòng này: {total_duration_with_w:.2f} giây")
input_file = "database8.tsv"
output_file = "clean_eng.tsv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    header = infile.readline()
    outfile.write(header)  # Ghi dòng tiêu đề
    keywords_to_exclude = ["aa","uo", "oo", "io","uc","oc", "up", "ip", "ap", "op", "ep", "aq", "oq", "iq", "uq", "eq", "as", "is",
                           "us", "es", "os", "ak", "ik", "uk", "ek", "ok", "ag", "ig", "ug", "eg", "og", "eu", "ou", "ei", "ad", "id", "ud", "ed", "od",
                           "ab", "ib", "ub", "eb", "ob", "iy", "ey", "oy", "av", "iv", "uv", "ev", "ov", "aw", "ax", "ix", "ux", "ex", "ox",
                           "ah", "ih", "uh", "eh", "oh", "al", "il", "ul", "el", "ol", "ar", "ir", "ur", "er", "or", "ii", "oo", "ee", "uu"
                           , "it", "ut", "ot","eu", "ou", "j", "w", "z", "f", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",]
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        text = parts[2].lower()
        if not any(keyword in text for keyword in keywords_to_exclude):
            outfile.write(line)

# import re

# def clean_text(text):
#     # Chuyển về chữ thường
#     text = text.lower()
#     # Bỏ các ký tự không phải chữ cái, số, hoặc khoảng trắng
#     text = re.sub(r'[^\w\s]', '', text)
#     # Thay nhiều khoảng trắng bằng 1
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# def process_tsv(input_path, output_path):
#     with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
#         for line in fin:
#             parts = line.strip().split('\t')
#             if len(parts) == 3:
#                 filename, duration, text = parts
#                 cleaned_text = clean_text(text)
#                 fout.write(f"{filename}\t{duration}\t{cleaned_text}\n")

# # Ví dụ sử dụng
# process_tsv(input_file, output_file)


