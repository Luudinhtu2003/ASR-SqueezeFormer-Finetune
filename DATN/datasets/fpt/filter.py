# def extract_references_each_line(input_txt_path, output_txt_path):
#     references = []
#     with open(input_txt_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.startswith("Reference : "):
#                 ref = line.replace("Reference : ", "").strip()
#                 references.append(ref)

#     with open(output_txt_path, 'w', encoding='utf-8') as f:
#         for ref in references:
#             f.write(ref + '\n')

# # === Chạy hàm ===
# input_txt = 'wer_above_0.30_eng_clean_8_train.txt'        # file chứa các Reference
# output_txt = 'references_only.txt'   # file mới chứa mỗi Reference 1 dòng

# extract_references_each_line(input_txt, output_txt)
#=================================================================
import re

def normalize(text):
    """Hàm chuẩn hóa văn bản: viết thường, xóa khoảng trắng thừa"""
    return re.sub(r'\s+', ' ', text.strip().lower())

def load_texts_to_remove(text_list_file):
    with open(text_list_file, 'r', encoding='utf-8') as f:
        return set(normalize(line) for line in f if line.strip())

def filter_tsv(tsv_input_path, tsv_output_path, texts_to_remove):
    with open(tsv_input_path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    header = lines[0]
    filtered_lines = [header]

    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        audio_path, duration, text = parts
        if normalize(text) not in texts_to_remove:
            filtered_lines.append(line)

    with open(tsv_output_path, 'w', encoding='utf-8') as fout:
        fout.writelines(filtered_lines)

# === CHẠY ===

text_list_file = 'references_only.txt'       # chứa các text cần loại (1 dòng 1 câu)
tsv_input = 'clean_eng_8_origin.tsv'              # file tsv gốc
tsv_output = 'Fclean_eng_8_origin.tsv'          # file sau khi loại

texts_to_remove = load_texts_to_remove(text_list_file)
filter_tsv(tsv_input, tsv_output, texts_to_remove)
