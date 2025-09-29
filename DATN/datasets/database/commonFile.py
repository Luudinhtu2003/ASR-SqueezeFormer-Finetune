with open('only_in_db8.tsv', 'r', encoding='utf-8') as f1, open('only_in_db8_filteredV2.tsv', 'r', encoding='utf-8') as f2:
    set1 = set(f1.readlines())
    set2 = set(f2.readlines())

# Lấy các dòng chỉ có trong only_in_db8_filteredV2.tsv mà không có trong only_in_db8.tsv
only_in_f1 = set1 - set2

with open('only_in_db8_filteredV3.tsv', 'w', encoding='utf-8') as fout:
    fout.writelines(only_in_f1)