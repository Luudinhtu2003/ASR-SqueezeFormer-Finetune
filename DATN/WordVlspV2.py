import glob
import re
import os
from underthesea import word_tokenize
from tqdm import tqdm  

# Đường dẫn chứa file văn bản
folder_path = r"F:\vlsp2020_train_set_02\*.txt"
text_files = glob.glob(folder_path)
unique_words = set()

# Biểu thức chính quy để loại bỏ ký tự đặc biệt (trừ chữ có dấu, số, dấu cách)
pattern = r"[^a-zA-ZÀ-ỹ0-9\s]"

# 🔹 Hàm đọc danh sách từ trong từ điển tiếng Việt
def load_vietnamese_dictionary(dictionary_path):
    vietnamese_words = set()
    if not os.path.exists(dictionary_path):
        print(f"⚠️ Không tìm thấy từ điển {dictionary_path}, sẽ tạo file trống!")
        open(dictionary_path, "w", encoding="utf-8").close()  # Tạo file trống nếu chưa có
    else:
        with open(dictionary_path, "r", encoding="utf-8") as f:
            vietnamese_words = {line.strip().lower() for line in f if line.strip()}
    return vietnamese_words

# 🔹 Đường dẫn file từ điển tiếng Việt (cần cập nhật nếu khác)
dictionary_path = "Viet74K.txt"
vietnamese_dictionary = load_vietnamese_dictionary(dictionary_path)

# 🔹 Hàm kiểm tra từ có trong từ điển không
def is_vietnamese(word):
    return word.lower() in vietnamese_dictionary

# Xử lý từng file
for file in tqdm(text_files, desc="Đang xử lý", unit="file"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            content = re.sub(pattern, "", content)  # Xóa ký tự đặc biệt
            words = word_tokenize(content)  # Tách từ
            mono_syllabic_words = {word for word in words if " " not in word and "-" not in word}  # Chỉ giữ từ đơn âm tiết
            unique_words.update(mono_syllabic_words)  # Thêm vào tập hợp (loại bỏ trùng lặp)
    except Exception as e:
        print(f"Lỗi khi đọc {file}: {e}")

# Phân loại từ
vietnamese_words = {word for word in unique_words if is_vietnamese(word)}
non_vietnamese_words = unique_words - vietnamese_words

# Sắp xếp danh sách từ
sorted_unique = sorted(unique_words)
sorted_vietnamese = sorted(vietnamese_words)
sorted_non_vietnamese = sorted(non_vietnamese_words)

# 🔹 Hàm lưu từ vào file
def save_words(filename, words):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{len(words)}\n")  # Ghi số lượng từ lên dòng đầu tiên
        for word in words:
            f.write(word + "\n")

# Lưu danh sách từ vào file
save_words("unique_monosyllabic_words.txt", sorted_unique)
save_words("vietnamese_words.txt", sorted_vietnamese)
save_words("non_vietnamese_words.txt", sorted_non_vietnamese)

# In kết quả
print(f"📌 Tổng số từ đơn âm tiết: {len(unique_words)}")
print(f"✅ Số từ tiếng Việt: {len(vietnamese_words)}")
print(f"❌ Số từ không phải tiếng Việt: {len(non_vietnamese_words)}")
