import os
import random
from glob import glob
import wave
import json

"""
Tạo file manifest cho tập dữ liệu speaker từ các file âm thanh và transcript
Tạo file manifest cho tập dữ liệu speaker
"""
# main_dir = r"F:\vlsp2020_train_set_02\speaker"  # Thay bằng đường dẫn thực tế
main_dir = r"F:\vivos\test\waves"  # Thay bằng đường dẫn thực tế #here
MANIFEST_DIR = "manifests"

os.makedirs (MANIFEST_DIR, exist_ok= True)

# Lấy danh sách tất cả folder con
all_subfolders = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
random.shuffle(all_subfolders)

folder_speaker = all_subfolders[0:]

from tqdm import tqdm

def get_wav_text_pairs(folder_list):
    data = []
    for folder in tqdm(folder_list, desc="Đang xử lý folder"):
        wav_files = glob(os.path.join(folder, '*.wav'))
        txt_files = glob(os.path.join(folder, '*.txt'))

        wav_map = {os.path.splitext(os.path.basename(f))[0]: f for f in wav_files}
        txt_map = {os.path.splitext(os.path.basename(f))[0]: f for f in txt_files}

        common_keys = set(wav_map.keys()) & set(txt_map.keys())
        for key in common_keys:
            wav_path = wav_map[key]
            txt_path = txt_map[key]

            # Không cần đọc nội dung file txt, chỉ lấy path
            data.append((wav_path, txt_path))

    return data


# Tạo các list
list_speaker = get_wav_text_pairs(folder_speaker)

# Hàm in ra n phần tử đầu tiên từ list
def print_list_sample(name, data_list, n=10):
    print(f"\n{name} ({len(data_list)} samples):")
    for i, (wav, txt) in enumerate(data_list[:n]):
        print(f"{i+1}. WAV: {wav} | TXT: {txt}")
splits = {
    'test_vivos': list_speaker, #here
    # 'test': test_list
}

for split_name, data_list in splits.items():
    print(f"\n{split_name.upper()} ({len(data_list)} samples):")
    for i, (wav, text) in enumerate(data_list[:5]):
        print(f"{i+1}. WAV: {wav} | TEXT: \"{text}\"")

#2. Hàm tính thời gian dài của file audio
def get_duration(wav_path):
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes() # Số lượng frame trong file
        rate = wf.getframerate()    #Tốc độ mẫu (sampling rate), ví dụ 16000 Hz
        return frames / float(rate)  #Thời gian = tổng só frame / tốc độ mẫu
    
#3. Tạo file manifest cho từng phần
for split_name, pairs in splits.items():
    manifest_path = os.path.join(MANIFEST_DIR, f"{split_name}_manifest.json")
    
    # Mở file để ghi manifest
    with open(manifest_path, "w", encoding="utf-8") as out_f:
        
        # Dùng tqdm để theo dõi tiến độ khi duyệt qua mỗi cặp wav và txt
        for wav_path, txt_path in tqdm(pairs, desc=f"Đang tạo manifest cho {split_name}", unit="sample"):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()  # Đọc nội dung và loại bỏ khoảng trắng đầu cuối
            except FileNotFoundError as e:
                print(f"Không tìm thấy file: {txt_path}")
                print(f"Lỗi chi tiết: {e}")
                text = ""  # Nếu không tìm thấy file, để text rỗng
            except Exception as e:
                print(f"Đã xảy ra lỗi khác khi đọc file {txt_path}: {e}")
                text = ""  # Nếu có lỗi khác, để text rỗng

            # Tính thời gian dài của file âm thanh
            duration = get_duration(wav_path)

            # Tạo thông tin của mỗi sample
            sample = {
                "audio_filepath": os.path.abspath(wav_path),  # Đường dẫn tuyệt đối đến file âm thanh
                "text": text,   # Transcript
                "duration": duration  # Thời gian của file âm thanh
            }

            # Ghi sample vào manifest
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"✔️ Created {split_name} manifest with {len(pairs)} samples.")

import json
import pandas as pd

def convert_jsonl_to_tsv(jsonl_path, tsv_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f_in, open(tsv_path, 'w', encoding='utf-8') as f_out:
        # Ghi header vào file TSV
        f_out.write("audio_filepath\tduration\ttext\n")
        
        for line in f_in:
            try:
                # Tải dữ liệu JSON từ mỗi dòng
                data = json.loads(line.strip())
                
                # Truy xuất các giá trị từ các khóa trong manifest
                audio_path = data.get("audio_filepath")
                duration = data.get("duration")
                text = data.get("text")
                
                # Kiểm tra xem các trường dữ liệu có tồn tại và hợp lệ không
                if audio_path and duration and text:
                    f_out.write(f"{audio_path}\t{duration}\t{text}\n")
                else:
                    print(f"Skipping line due to missing data: {line.strip()}")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"Error processing line: {line.strip()}. Error: {e}")

                
                

# convert_jsonl_to_tsv(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\manifests\speaker_manifest.json", "speaker.tsv")
convert_jsonl_to_tsv(r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\manifests\test_vivos_manifest.json", "test_vivos.tsv")


#Chuyển text về chữ thường trong file TSV
# Đọc file TSV #here nếu là vivos
# df = pd.read_csv('test_vivos.tsv', sep='\t')

# # Chuyển cột 'text' thành chữ thường
# df['text'] = df['text'].str.lower()

# # Lưu lại file mới (ghi đè hoặc lưu tên khác)
# df.to_csv('test_vivos.tsv', sep='\t', index=False)