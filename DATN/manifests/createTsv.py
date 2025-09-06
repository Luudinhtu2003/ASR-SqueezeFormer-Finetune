import json


def convert_manifest_to_tsv(manifest_path, tsv_path):
        with open(manifest_path, 'r', encoding='utf-8') as f_in, open(tsv_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                data = json.loads(line.strip())
                audio_path = data.get("audio_filepath") or data.get("audio") or data.get("wav")
                text = data.get("text") or data.get("label") or data.get("transcript")
                if audio_path and text:
                    f_out.write(f"{audio_path}\t{text}\n")
                
convert_manifest_to_tsv("train_manifest_processed.json", "train.tsv")
convert_manifest_to_tsv("test_manifest_processed.json", "test.tsv")