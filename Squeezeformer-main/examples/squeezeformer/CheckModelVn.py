import sentencepiece as spm

# Tải mô hình SentencePiece đã huấn luyện
sp = spm.SentencePieceProcessor()
sp.load('vn_model.model')  # Đảm bảo rằng đường dẫn đúng tới file vi_model.model
# Lấy kích thước vocab
# Lấy toàn bộ token
vocab_size = sp.get_piece_size()
tokens = [sp.id_to_piece(i) for i in range(vocab_size)]

# Sắp xếp token theo bảng chữ cái
tokens_sorted = sorted(tokens)

# In ra
print(f"Vocab size: {vocab_size}")
print("Sorted tokens:")
for i, token in enumerate(tokens_sorted):
    print(f"{i}: {token}")
# Văn bản cần mã hóa và giải mã
text = "thời tiết hôm nay thật đẹp chỉ còn anh lang thang chốn cũ lòng quặn đau nước mắt ướt nhòa người mà anh đã từng nguyện yêu thương"

# Mã hóa (Encode) văn bản thành các token IDs
token_ids = sp.encode(text, out_type=int)
print(f"Token IDs: {token_ids}")

# Giải mã (Decode) token IDs trở lại văn bản
decoded_text = sp.decode(token_ids)
print(f"Decoded text: {decoded_text}")
