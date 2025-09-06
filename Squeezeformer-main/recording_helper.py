"""
Thử record xem thế nào
"""
import pyaudio
import numpy as np
import time
import keyboard
import librosa
import librosa.display
import matplotlib.pyplot as plt

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_SECONDS = 5

p = pyaudio.PyAudio()

def record_while_key_pressed(key='r'):
    print(f"Nhấn giữ phím '{key}' để ghi âm (tối đa {MAX_SECONDS} giây)...")

    while not keyboard.is_pressed(key):
        pass

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    print("Đang ghi âm...")

    frames = []
    start_time = time.time()
    
    while keyboard.is_pressed(key) and (time.time() - start_time < MAX_SECONDS):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    print("Ghi âm dừng lại.")

    stream.stop_stream()
    stream.close()

    return np.frombuffer(b''.join(frames), dtype=np.int16)

def show_mel_spectrogram(audio, sr):
    # Chuyển đổi sang float32 nếu cần
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # Tính Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=80)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def terminate():
    p.terminate()

if __name__ == "__main__":
    audio_data = record_while_key_pressed()
    terminate()
    print(f"Độ dài tín hiệu thu: {len(audio_data)} mẫu.")
    show_mel_spectrogram(audio_data, RATE)
