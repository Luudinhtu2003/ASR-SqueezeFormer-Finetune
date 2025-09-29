import csv

input_file = 'only_in_db8.tsv'
output_file = 'only_in_db8_filterdd.tsv'

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8', newline='') as fout:
    reader = csv.DictReader(fin, delimiter='\t')
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter='\t')
    writer.writeheader()
    for row in reader:
        try:
            duration = float(row['duration'])
            text = row['text']
            word_count = len(text.strip().split())
            if duration >= 1.0 and word_count > 30 and word_count <= 40:
                writer.writerow(row)
        except Exception as e:
            continue  # Bỏ qua dòng lỗi