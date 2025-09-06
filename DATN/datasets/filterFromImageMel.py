"""
File này dùng để lọc bỏ những file có Mel Spectrogram 
không có trong hình ảnh mel spectrogram đã tạo (do đã bỏ đi những ảnh xấu).
"""
import os

# Đường dẫn tới folder chứa ảnh mel spectrogram
# mel_folder = r"F:\Luu_Dinh_Tu\Project_2\Squeezeformer-main\mel_spectrogram_fpt" #here
mel_folder = r"F:\Luu_Dinh_Tu\Project_2\Squeezeformer-main\mel_spectrogram_database"

# Lấy danh sách tên file (không đuôi) trong folder mel
mel_files = {os.path.splitext(f)[0] for f in os.listdir(mel_folder) if os.path.isfile(os.path.join(mel_folder, f))}

# Đọc file TSV và ghi ra những dòng còn mel tương ứng
with open("database.tsv", "r", encoding="utf-8") as infile, open("databaseFMel.tsv", "w", encoding="utf-8") as outfile: #here
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            file_path = parts[0]
            base_name = os.path.splitext(os.path.basename(file_path))[0]  # Lấy tên file không đuôi
            if base_name in mel_files:
                outfile.write(line)
