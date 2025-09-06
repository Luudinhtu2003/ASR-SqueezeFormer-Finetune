import sys
import os
import numpy as np

# Thêm thư mục gốc của dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import các module
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
import matplotlib.pyplot as plt
import math

speech_config = {
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "feature_type": "log_mel_spectrogram",
    "delta": False,
    "delta_delta": False,
    "pitch": False,
    "normalize_signal": True,
    "normalize_feature": True,
    "normalize_per_frame": False,
    "preemphasis": 0.97,  # tuỳ chọn
    "top_db": 80.0,
    "center": True
}

featurizer = TFSpeechFeaturizer(speech_config)
import librosa
import tensorflow as tf

signal, sr = librosa.load("database_sa1_Jan08_Mar19_cleaned_utt_0000000005-1.wav", sr=16000)

#Trích xuất đặc trưng:
features = featurizer.extract(signal)
print("Kich thuoc trich xuat dac trung thu cong")
print(features.shape)
features = np.squeeze(features, axis=-1)  # shape [T, 80]

# Vẽ log-Mel spectrogram
plt.figure(figsize=(12, 4))
plt.imshow(features.T, aspect='auto', origin='lower', interpolation='none')
plt.title("Log-Mel Spectrogram")
plt.xlabel("Time frames")
plt.ylabel("Mel frequency bins")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()

# Load file WAV
file_path = "database_sa1_Jan08_Mar19_cleaned_utt_0000000005-1.wav"
y, sr = librosa.load(file_path, sr=16000)  # sr=16000 để đồng nhất với config

# Config giống với TFSpeechFeaturizer
frame_ms = 25
stride_ms = 10
n_fft = 2 ** int(np.ceil(np.log2(sr * frame_ms / 1000)))  # giống như featurizer.nfft
hop_length = int(sr * stride_ms / 1000)
win_length = int(sr * frame_ms / 1000)
n_mels = 80

# Trích xuất log-Mel spectrogram
S = librosa.feature.melspectrogram(y=y,
                                   sr=sr,
                                   n_fft=n_fft,
                                   hop_length=hop_length,
                                   win_length=win_length,
                                   n_mels=n_mels,
                                   power=2.0)

log_S = librosa.power_to_db(S, ref=np.max)

# Vẽ log-Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_S, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='mel', cmap='magma')
plt.title("Log-Mel Spectrogram (librosa)")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

print("Librosa feature shape:", log_S.T.shape)  # (T, 80)

from src.featurizers.text_featurizers import SentencePieceFeaturizer

# Cấu hình với model path
decoder_config = {
    "vocabulary": "F:\Luu_Dinh_Tu\Project_2\Squeezeformer-main\examples\squeezeformer\spm_model.model"
}

# Khởi tạo featurizer
featurizer = SentencePieceFeaturizer(decoder_config)

text = "Đảng cộng sản việt nam muôn năm"
ids_tensor = featurizer.extract(text)
print("IDs:", ids_tensor.numpy())  # ví dụ: [132, 453, 991, ...]
# Giả sử đây là kết quả model predict
decoded_prediction = ids_tensor  # hoặc một tensor giống vậy

# Cần đưa vào dạng [1, None] nếu decode 1 câu
decoded = featurizer.iextract(tf.expand_dims(decoded_prediction, axis=0))
print("Decoded:", decoded.numpy()[0].decode("utf-8"))




