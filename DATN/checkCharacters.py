import pandas as pd
from collections import Counter

# Bước 1: Đọc file .tsv
df = pd.read_csv(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\speaker_vivos\Fspeaker_vivos_train_8.tsv", sep='\t')  # Thay bằng đường dẫn file thật

# Bước 2: Gộp toàn bộ text thành một chuỗi
all_text = ''.join(df['text'].astype(str))

# Bước 3: Đếm số lần xuất hiện các ký tự
char_counter = Counter(all_text)

# Bước 4: Danh sách labels
labels = ['', '<s>', '</s>', ' ', 'n', 'h', 't', 'i', 'c', 'g', 'a', 'm', 'u', 'đ', 'à', 'o', 'ư', 'v', 'l', 'r', 'á', 'y', 'b', 'p', 'ô', 'k', 's', 'ó', 'ế', 'ạ', 'ộ', 'ờ', 'ệ', 'ả', 'ê', 'ì', 'd', 'â', 'ố', 'ớ', 'ấ', 'ơ', 'ề', 'q', 'ủ', 'ể', 'ă', 'ị', 'ợ', 'í', 'ậ', 'e', 'x', 'ầ', 'ự', 'ú', 'ữ', 'ọ', 'ứ', 'ã', 'ở', 'ồ', 'ụ', 'ắ', 'ừ', 'ổ', 'ò', 'ũ', 'ù', 'ặ', 'ý', 'ỉ', 'ẽ', 'ỏ', 'ử', 'ằ', 'é', 'ĩ', 'ễ', 'ẩ', 'ẫ', 'ỗ', 'ẹ', 'ỹ', 'ẻ', 'ỳ', 'è', 'õ', 'ỡ', 'ẳ']

# Chỉ lấy các ký tự thực sự (loại bỏ '', <s>, </s>)
filtered_labels = [ch for ch in labels if len(ch) == 1]

# Bước 5: Ghi kết quả ra file .txt
with open("thongke_ky_tu.txt", "w", encoding="utf-8") as f:
    for ch in filtered_labels:
        count = char_counter.get(ch, 0)
        # Ghi theo định dạng: 'ký tự': số lần
        if ch == ' ':
            f.write(f"'(space)': {count}\n")
        else:
            f.write(f"'{ch}': {count}\n")

print("Đã lưu thống kê vào file 'thongke_ky_tu.txt'")
