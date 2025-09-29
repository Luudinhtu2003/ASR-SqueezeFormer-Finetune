"""
Kiểm tra phân bố của cột 'duration' trong tập dữ liệu và hiển thị biểu đồ phân bố
Dùng để kiểm tra số lượng file có duration lớn hơn x giây và tổng duration của các file này
Muốn dùng lại chỉ cần thay đổi file_path and giá trị muốn so sánh
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file TSV
# file_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\train_all_processed.tsv"  # Thay đổi đường dẫn đến file TSV của bạn
file_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivosV3\fpt_valid.tsv"  # here
df = pd.read_csv(file_path, sep='\t')

# Kiểm tra đầu dữ liệu (để đảm bảo file được đọc đúng)
print(df.head())

# Tính số lượng file có duration lớn hơn 10 giây
# count_large_files = len(df[df['duration'] > 1])
count_large_files = len(df[(df['duration'] <= 1)])
total_duration = df['duration'].sum()
# short_duration = df[df['duration'] <= 8]['duration'].sum()
short_duration = df[(df['duration'] >= 0) & (df['duration'] <= 8)]['duration'].sum()


# Vẽ biểu đồ phân bố cho cột 'duration'
plt.figure(figsize=(12, 6))
sns.histplot(df['duration'], kde=True, bins=30, color='skyblue')  # Vẽ histogram với đường cong KDE

# Tùy chỉnh tiêu đề và nhãn
plt.title('Phân bố của duration tập train VLSP', fontsize=16)
plt.xlabel('Duration (giây)', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    else:
        return f"{minutes}m {secs}s"
total_str = format_duration(total_duration)
short_str = format_duration(short_duration)
# Hiển thị số lượng các file có duration > 10s ở góc phải
plt.text(0.95, 0.95, f"Files > 8s: {count_large_files}\n"
         f"Total Duration: {total_str}s\n"
         f"≤ 8s Duration: {short_str}s", 
         fontsize=12, color='red', ha='right', va='top', transform=plt.gca().transAxes)

# Hiển thị biểu đồ
plt.show()
