import pandas as pd
import os

def check_tsv_file(file_path):
    print(f"🔍 Đang kiểm tra file: {file_path}")

    if not os.path.exists(file_path):
        print("❌ File không tồn tại.")
        return

    try:
        df = pd.read_csv(file_path, sep="\t", dtype=str)
    except Exception as e:
        print(f"❌ Không thể đọc file: {e}")
        return

    print(f"✅ Đọc thành công, số dòng: {len(df)}")

    # Kiểm tra các cột cần thiết
    required_columns = {"audio_filepath", "duration", "text"}
    if not required_columns.issubset(df.columns):
        print(f"❌ Thiếu cột: {required_columns - set(df.columns)}")
        return

    # Kiểm tra dòng thiếu dữ liệu
    missing_rows = df[df.isnull().any(axis=1)]
    if not missing_rows.empty:
        print(f"❌ Có {len(missing_rows)} dòng bị thiếu dữ liệu:")
        print(missing_rows)
    else:
        print("✅ Không có dòng nào thiếu dữ liệu.")

    # Kiểm tra kiểu dữ liệu bất thường (list, dict)
    for col in ["audio_filepath", "duration", "text"]:
        complex_rows = df[df[col].apply(lambda x: isinstance(x, (list, dict)))]
        if not complex_rows.empty:
            print(f"❌ Cột '{col}' có giá trị kiểu list hoặc dict:")
            print(complex_rows)

    # Thử chuyển sang dataset TensorFlow
    try:
        import tensorflow as tf
        dataset_dict = {
            "audio_filepath": df["audio_filepath"].tolist(),
            "duration": df["duration"].tolist(),
            "text": df["text"].tolist()
        }
        _ = tf.data.Dataset.from_tensor_slices(dataset_dict)
        print("✅ Dữ liệu có thể dùng với tf.data.Dataset.from_tensor_slices.")
    except Exception as e:
        print(f"❌ Lỗi khi tạo Dataset TensorFlow: {e}")

# 👉 Gọi hàm với file thật
check_tsv_file("Fspeaker_vivos_fpt_train_8.tsv")
