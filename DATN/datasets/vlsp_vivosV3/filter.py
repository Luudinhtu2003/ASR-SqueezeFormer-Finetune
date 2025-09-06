input_file = "wer_above_0.10_fpt_65_trainV2.txt"    # Đổi tên file đầu vào tại đây
output_file = "filter_fpt_train.txt"    # Tên file đầu ra

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    lines = infile.readlines()
    for i in range(len(lines)):
        if lines[i].startswith("WER"):
            try:
                wer_value = float(lines[i].split(":")[1].strip())
                if wer_value >= 0.6:
                    reference_line = lines[i - 2]
                    if reference_line.startswith("Reference :"):
                        text = reference_line.split(":", 1)[1].strip()
                        outfile.write(text + "\n")
            except:
                continue
