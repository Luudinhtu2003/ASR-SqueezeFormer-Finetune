import matplotlib.pyplot as plt
import ast

# Đọc dữ liệu từ file .txt
with open(r"F:\Luu_Dinh_Tu\Project_2\Squeezeformer-main\loss_plots10\log_metrics.txt", "r") as f:
    lines = f.readlines()

valid_loss = ast.literal_eval(lines[0].strip())
train_loss = ast.literal_eval(lines[1].strip())
test_wer   = ast.literal_eval(lines[2].strip())
valid_wer  = ast.literal_eval(lines[3].strip())

epochs = list(range(1, len(valid_loss) + 1))
# Cài đặt font to và đẹp hơn
plt.rcParams.update({
    'font.size': 18,             # font chung
    'axes.titlesize': 20,        # tiêu đề trục
    'axes.labelsize': 18,        # tên trục
    'xtick.labelsize': 16,       # số trên trục x
    'ytick.labelsize': 16,       # số trên trục y
    'legend.fontsize': 16        # chú thích
})

# Vẽ biểu đồ
plt.figure(figsize=(15, 6))

# --- Biểu đồ LOSS ---
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2.5, marker='o', markersize=10)
plt.plot(epochs, valid_loss, label='Valid Loss', color='#2ca02c', linewidth=2.5, marker='s', markersize=10)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid(True)

# --- Biểu đồ WER ---
plt.subplot(1, 2, 2)
plt.plot(epochs, test_wer, label='Test WER', color='#d62728', linewidth=2.5, marker='^', markersize=10)
plt.plot(epochs, valid_wer, label='Valid WER', color='#ff7f0e', linewidth=2.5, marker='D', markersize=10)
plt.xlabel('Epoch')
plt.ylabel('WER')
plt.title('WER per Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()