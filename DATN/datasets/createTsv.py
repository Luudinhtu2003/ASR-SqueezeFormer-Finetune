"""
File này được sử dụng để chuyển đổi file JSONL thành file TSV.
"""
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


# Chuyển text về chữ thường trong file TSV
# Đọc file TSV #here nếu là vivos
df = pd.read_csv('test_vivos.tsv', sep='\t')

# Chuyển cột 'text' thành chữ thường
df['text'] = df['text'].str.lower()

# Lưu lại file mới (ghi đè hoặc lưu tên khác)
df.to_csv('test_vivos.tsv', sep='\t', index=False)