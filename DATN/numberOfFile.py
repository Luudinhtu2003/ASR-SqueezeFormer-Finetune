import os
import time
import sys

def load_non_vietnamese_words(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file]
    except Exception as e:
        print(f"Lỗi khi đọc file danh sách từ: {e}")
        return []

def count_txt_files_with_foreign_words(folder_path, non_vietnamese_words, threshold=4):
    count = 0
    start_time = time.time()
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    total_files = len(file_list)
    
    if total_files == 0:
        print("Không có file .txt nào trong thư mục.")
        return 0
    
    print(f"Tổng số file cần xử lý: {total_files}")
    
    for idx, filename in enumerate(file_list, start=1):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                word_count = sum(1 for word in non_vietnamese_words if word in text)
                if word_count >= threshold:
                    count += 1
                    print(f"File: {filename} chứa {word_count} từ không phải tiếng Việt")
        except Exception as e:
            print(f"Lỗi khi đọc {filename}: {e}")
        
        # Ước tính thời gian hoàn thành
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / idx) * total_files
        remaining_time = estimated_total_time - elapsed_time
        sys.stdout.write(f"\rĐang xử lý {idx}/{total_files} file... Ước tính tổng thời gian: {estimated_total_time:.2f} giây, còn lại {remaining_time:.2f} giây")
        sys.stdout.flush()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nThời gian hoàn thành toàn bộ quá trình: {elapsed_time:.2f} giây")
    
    return count



# Thay đổi đường dẫn folder_path và file chứa danh sách từ
folder_path = r"F:\vlsp2020_train_set_02"
words_file_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\non_vietnamese_words.txt"

# Load danh sách từ không phải tiếng Việt từ file
non_vietnamese_words = load_non_vietnamese_words(words_file_path)

result = count_txt_files_with_foreign_words(folder_path, non_vietnamese_words)
print(f"Số file .txt chứa ít nhất 4 từ không phải tiếng Việt: {result}")
