import os
from pydub import AudioSegment
from tqdm import tqdm

# # Thư mục chứa file .mp3 gốc
# mp3_folder = r"F:\archive\mp3"

# # Thư mục để lưu file .wav sau chuyển đổi
# wav_folder = r"F:\archive\mp3V2"
# os.makedirs(wav_folder, exist_ok=True)

# # Danh sách file .mp3 trong thư mục
# mp3_files = [f for f in os.listdir(mp3_folder) if f.lower().endswith(".mp3")]

# # Duyệt từng file có tiến trình
# for mp3_file in tqdm(mp3_files, desc="Converting MP3 to WAV"):
#     mp3_path = os.path.join(mp3_folder, mp3_file)
#     wav_filename = os.path.splitext(mp3_file)[0] + ".wav"
#     wav_path = os.path.join(wav_folder, wav_filename)

#     # Load mp3 và convert
#     sound = AudioSegment.from_mp3(mp3_path)
#     sound = sound.set_frame_rate(16000)  # đặt sample rate = 16000 Hz
#     sound.export(wav_path, format="wav")
# import os
# from pydub import AudioSegment

# mp3_folder = r"F:\archive\mp3" # <- Đổi đường dẫn đến folder chứa MP3
# output_wav_folder = r"F:\archive\mp3V2"         # <- Nơi bạn muốn lưu WAV
# error_log_path = "error_log.txt"               # File để ghi các lỗi

# os.makedirs(output_wav_folder, exist_ok=True)

# with open(error_log_path, "w", encoding="utf-8") as log_file:
#     for filename in os.listdir(mp3_folder):
#         if filename.lower().endswith(".mp3"):
#             mp3_path = os.path.join(mp3_folder, filename)
#             wav_path = os.path.join(output_wav_folder, os.path.splitext(filename)[0] + ".wav")

#             try:
#                 sound = AudioSegment.from_mp3(mp3_path)
#                 # Ép sample rate về 16000 Hz
#                 sound = sound.set_frame_rate(16000)
#                 sound.export(wav_path, format="wav")
#                 print(f"✅ Converted: {filename} → sample rate 16000 Hz")
#             except Exception as e:
#                 print(f"❌ Error in: {filename}")
#                 log_file.write(f"{filename}: {e}\n") 
mp3_dir = r"F:\archive\mp3"
wav_dir = r"F:\archive\mp3V2" 

# Tạo danh sách file wav đã tồn tại (không tính đuôi .wav)
existing_wav_files = {os.path.splitext(f)[0] for f in os.listdir(wav_dir) if f.endswith('.wav')}

# Lặp qua các file mp3
for filename in os.listdir(mp3_dir):
    if filename.endswith('.mp3'):
        name = os.path.splitext(filename)[0]
        if name not in existing_wav_files:
            mp3_path = os.path.join(mp3_dir, filename)
            wav_path = os.path.join(wav_dir, name + '.wav')
            try:
                audio = AudioSegment.from_mp3(mp3_path)
                audio = audio.set_frame_rate(16000)  # ✅ Đặt sample rate 16kHz
                audio.export(wav_path, format='wav')
                print(f'Đã chuyển {filename} -> {name}.wav (16000Hz)')
            except Exception as e:
                print(f'Lỗi khi chuyển {filename}: {e}')
