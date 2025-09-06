"""
Trích xuất toàn bộ các file dựa vào file tsv lưu ra ảnh
Dùng lại chỉ cần thay file tsv chứa đường dẫn file âm thanh và folder lưu ảnh
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Trichs xuất mel spectrogram từ file âm thanh và lưu dưới dạng ảnh
# Config
speech_config = {
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "top_db": 80.0,
    "center": True,
    "normalize_signal": True,
    "preemphasis": 0.97,
}

# Đường dẫn file .tsv và thư mục lưu ảnh
tsv_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\database.tsv" #here
output_folder = "mel_spectrogram_database"
os.makedirs(output_folder, exist_ok=True)

# Đọc danh sách audio paths từ .tsv
df = pd.read_csv(tsv_path, sep="\t")
audio_paths = df['audio_filepath'].tolist()  # Giả sử cột chứa đường dẫn tên là "path"

# Hàm chuẩn hóa tín hiệu
def normalize_signal(signal):
    return signal / max(abs(signal)) if max(abs(signal)) != 0 else signal

# Xử lý từng file WAV với thanh tiến trình
for path in tqdm(audio_paths, desc="Processing audio files"):
    try:
        # Load audio
        y, sr = librosa.load(path, sr=speech_config["sample_rate"])

        # Normalize tín hiệu nếu cần
        if speech_config["normalize_signal"]:
            y = normalize_signal(y)

        # Preemphasis
        if speech_config["preemphasis"]:
            y = librosa.effects.preemphasis(y, coef=speech_config["preemphasis"])

        # Trích xuất mel spectrogram
        hop_length = int(sr * speech_config["stride_ms"] / 1000)
        win_length = int(sr * speech_config["frame_ms"] / 1000)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=speech_config["num_feature_bins"],
            hop_length=hop_length,
            win_length=win_length,
            center=speech_config["center"]
        )
        S_dB = librosa.power_to_db(S, top_db=speech_config["top_db"])

        # Plot và lưu ảnh
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.axis('off')  # tắt trục nếu chỉ muốn hình
        filename = os.path.splitext(os.path.basename(path))[0] + ".png"
        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Lỗi xử lý file {path}: {e}")

print("✅ Hoàn tất trích xuất và lưu ảnh Mel spectrogram.")
