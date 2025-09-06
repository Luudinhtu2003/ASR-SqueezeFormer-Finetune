import sentencepiece as spm

# Đường dẫn đến file dữ liệu văn bản
corpus_file = 'createModel.txt'

# Đường dẫn lưu model và vocab
model_prefix = '93spModel'

# Số lượng token tối đa (ví dụ: 8000 token)
# vocab_size = 128
vocab_size = 93

# Huấn luyện mô hình SentencePiece
# spm.SentencePieceTrainer.train(input=corpus_file, model_prefix=model_prefix, vocab_size=vocab_size, model_type='bpe')
spm.SentencePieceTrainer.train(
    input=corpus_file,
    model_prefix=model_prefix,
    vocab_size=93,  # hoặc lớn hơn
    model_type='bpe',
    character_coverage=1.0
)
print("Model và vocab đã được tạo!")