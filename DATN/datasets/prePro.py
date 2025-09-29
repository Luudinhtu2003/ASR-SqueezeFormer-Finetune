import pandas as pd
import os

# Đọc dữ liệu từ file TSV
file_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\fpt\fpt_65_test.tsv"  # Thay đổi đường dẫn đến file TSV của bạn
df = pd.read_csv(file_path, sep='\t')

# Lọc các dòng có 'duration' <= 8 giây
filtered_df = df[df['duration'] <= 5]



# Lấy thư mục của file gốc
folder_path = os.path.dirname(file_path)

# Đặt tên cho file TSV mới và lưu vào thư mục gốc
filtered_file_path = os.path.join(folder_path, "fpt_65_testV2.tsv")

# Lưu kết quả vào file TSV mới
filtered_df.to_csv(filtered_file_path, sep='\t', index=False)

print(f"File đã được lưu tại {filtered_file_path}")
