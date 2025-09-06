"""
File để dòn tất cả file của mỗi người vào folder của họ từ dataset mẫu cung cấp,
Chỉ chứa toàn bộ wav, text và phân biệt người dựa vào tên file.
"""
import os
import shutil

# Đường dẫn tới thư mục chứa các file
source_dir = r"F:\vlsp2020_train_set_02\speaker"  # Thay đổi đường dẫn này theo thư mục của bạn

# Thư mục đích nơi bạn muốn lưu các thư mục speaker_id
destination_dir = r"F:\vlsp2020_train_set_02\destination"  # Thay bằng đường dẫn đến thư mục đích

# Lấy danh sách các file trong thư mục
files = os.listdir(source_dir)

# Tạo một dictionary để nhóm các file theo speaker_id
speaker_files = {}

# Duyệt qua từng file và nhóm chúng theo speaker_id
for file in files:
    # Lấy ID speaker: phần từ 'speaker_' đến phần trước dấu '_' hoặc '-'
    if 'speaker_' in file:
        # Lấy phần trước dấu '_' hoặc dấu '-' sau 'speaker_'
        speaker_id = file.split('speaker_')[1].split('-')[0].split('_')[0]
        
        # Nhóm các file theo speaker_id
        if speaker_id not in speaker_files:
            speaker_files[speaker_id] = []
        speaker_files[speaker_id].append(file)

# Di chuyển các file vào thư mục tương ứng với mỗi speaker_id
for speaker_id, file_list in speaker_files.items():
    # Tạo thư mục cho speaker trong thư mục nguồn nếu chưa tồn tại
    speaker_folder = os.path.join(source_dir, speaker_id)
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)

    # Di chuyển các file có chung speaker_id vào thư mục tương ứng
    for file in file_list:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(speaker_folder, file)
        shutil.move(source_path, destination_path)

# Sao chép toàn bộ thư mục speaker_id vào thư mục đích mà không thay đổi gì
for speaker_id in speaker_files.keys():
    # Đường dẫn thư mục nguồn chứa các file của speaker_id
    speaker_folder_source = os.path.join(source_dir, speaker_id)

    # Đường dẫn thư mục đích
    speaker_folder_destination = os.path.join(destination_dir, speaker_id)

    # Kiểm tra nếu thư mục đích chưa tồn tại, tạo nó
    if not os.path.exists(speaker_folder_destination):
        shutil.copytree(speaker_folder_source, speaker_folder_destination)

print("Các file và thư mục đã được sao chép sang thư mục đích.")
