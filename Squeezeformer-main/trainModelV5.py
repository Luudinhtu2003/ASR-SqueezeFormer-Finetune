"""
folder này để train vlsp, vivos, đổi sang 93 ký tự duy nhất
"""
import os
from tqdm import tqdm
import argparse
from scipy.special import softmax
import matplotlib.pyplot as plt
from datasets import load_metric
import tensorflow as tf
from tensorflow.keras import backend as K
import random
import numpy as np

from src.configs.config import Config
from src.datasets.asr_dataset import ASRSliceDataset
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.utils import env_util

wer_metric = load_metric("wer")  # chỉ cần load 1 lần
logger = env_util.setup_environment()

# Kiểm tra và thiết lập bộ nhớ GPU trước khi bất kỳ phép toán nào xảy ra.
physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

os.makedirs("loss_plots7", exist_ok=True) #here
# Parse command line
def parse_arguments():
    parser = argparse.ArgumentParser(prog="Conformer Testing")

    parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
    parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")
    parser.add_argument("--device", type=int, default=0, help="Device's id to run test on")
    parser.add_argument("--cpu", default=False, action="store_true", help="Whether to only use cpu")
    parser.add_argument("--saved", type=str, default=None, help="Path to saved model")
    parser.add_argument("--output", type=str, default=None, help="Result filepath")

    # Dataset arguments
    parser.add_argument("--bs", type=int, default=None, help="Test batch size")
    #parser.add_argument("--dataset_path", type=str, help="path to the tsv manifest files")
    #parser.add_argument("--dataset", type=str, default="test_other", 
     #                   choices=["dev_clean", "dev_other", "test_clean", "test_other"], help="Testing dataset")
    parser.add_argument("--input_padding", type=int, default=803)
    #parser.add_argument("--label_padding", type=int, default=132)
    parser.add_argument("--label_padding", type=int, default=195)#here
    # Architecture arguments
    parser.add_argument("--fixed_arch", default=None, help="force fixed architecture")

    # Decoding arguments
    parser.add_argument("--beam_size", type=int, default=None, help="ctc beam size")

    args = parser.parse_args()
    return args
augmentation_config = {
    "time_masking": {
        "num_masks": 5,
        "p_upperbound": 0.05
    },
    "freq_masking": {
        "num_masks": 2,
        "mask_factor": 27
    }
}

NUM_LAYERS_IN_BLOCK = 4
def parse_fixed_arch(args):
    parsed_arch = args.fixed_arch.split('|')
    i, rep = 0, 1
    fixed_arch = []
    while i < len(parsed_arch):
        if parsed_arch[i].isnumeric():
            rep = int(parsed_arch[i])
        else:
            block = parsed_arch[i].split(',')
            assert len(block) == NUM_LAYERS_IN_BLOCK
            for _ in range(rep):
                fixed_arch.append(block)
            rep = 1
        i += 1
    return fixed_arch
def parse_fixed_arch_from_yaml(fixed_arch_list, num_blocks=16, num_layers_in_block=4):
    assert len(fixed_arch_list) == num_layers_in_block, \
        f"Expected {num_layers_in_block} layers per block, got {len(fixed_arch_list)}"
    return [fixed_arch_list.copy() for _ in range(num_blocks)]

args = parse_arguments()

config = Config(args.config)

NUM_BLOCKS = config.model_config['encoder_num_blocks']

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})
env_util.setup_devices([args.device], cpu=args.cpu)


speech_featurizer = TFSpeechFeaturizer(config.speech_config)

logger.info("Use SentencePiece ...")
text_featurizer = SentencePieceFeaturizer(config.decoder_config)

seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Parse fixed architecture
if args.fixed_arch is not None:
    #fixed_arch = parse_fixed_arch(args)
    fixed_arch = parse_fixed_arch_from_yaml(config.model_config['encoder_fixed_arch'])

    if len(fixed_arch) != NUM_BLOCKS:
        logger.warn(
            f"encoder_num_blocks={config.model_config['encoder_num_blocks']} is " \
            f"different from len(fixed_arch) = {len(fixed_arch)}." \
        )
        logger.warn(f"Changing `encoder_num_blocks` to {len(fixed_arch)}")
        config.model_config['encoder_num_blocks'] = len(fixed_arch)
    logger.info(f"Changing fixed arch: {fixed_arch}")
    config.model_config['encoder_fixed_arch'] = fixed_arch

dataset_train_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivosV3\train.tsv"
dataset_val_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivosV3\valid.tsv" #here
dataset_test_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\vlsp_vivosV3\test.tsv"

if dataset_train_path is not None:
    logger.info(f"dataset train: at {dataset_train_path}")
    config.learning_config.train_dataset_config.data_paths = [dataset_train_path]
else:
    raise ValueError("specify the manifest file path using --dataset_train_path")

if dataset_val_path is not None:
    logger.info(f"dataset valid: at {dataset_val_path}")
    config.learning_config.eval_dataset_config.data_paths = [dataset_val_path]
else:
    raise ValueError("specify the manifest file path using --dataset_valid_path")

if dataset_test_path is not None:
    logger.info(f"dataset test: at {dataset_test_path}")
    config.learning_config.test_dataset_config.data_paths = [dataset_test_path]
else:
    raise ValueError("specify the manifest file path using --dataset_valid_path")


train_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.train_dataset_config) # Load parameters for train
)
valid_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.eval_dataset_config) #Load parameters for valid
)

test_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.test_dataset_config) #Load parameters for test
)
def build_conformer(config, vocab_size, featurizer_shape, augmentation_config=None):
    model = ConformerCtc(
        **config.model_config, 
        vocabulary_size=vocab_size,
        augmentation_config=augmentation_config, 
    )
    model.make(featurizer_shape)
    return model

conformer = build_conformer(config, text_featurizer.num_classes, speech_featurizer.shape)
temp_model = build_conformer(config, 128, speech_featurizer.shape)

# Load weights
if args.saved:
    temp_model.load_weights(args.saved, by_name=True)
    conformer.encoder.set_weights(temp_model.encoder.get_weights())
    logger.info(f"Weights loaded from {args.saved}")
else:
    logger.warning("Model initialized randomly. Consider using --saved to assign checkpoint.")
# Freeze encoder
del temp_model

conformer.decoder.trainable = True

encoder_layer = conformer.get_layer("conformer_encoder")


# Lặp qua tất cả các layers bên trong encoder
for i, layer in enumerate(encoder_layer.layers):
    #if layer.name in ["conformer_encoder_block_14", "conformer_encoder_block_15", "dense"]:#here
    if layer.name in ["conformer_encoder_block_6","conformer_encoder_block_14", "conformer_encoder_block_15", "dense"]: #here
        layer.trainable = True  # Unfreeze layer 14 và 15
    else:
        layer.trainable = False  # Freeze các layer còn lại

#conformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2))
# total_steps = 160 * 593  # 94880

# lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
#     initial_learning_rate=1e-3,
#     first_decay_steps= 5 * 593,     # ví dụ: chu kỳ đầu dài 1/10 tổng số bước
#     t_mul=2.0,                   # chu kỳ sau gấp đôi chu kỳ trước
#     m_mul=0.9                    # biên độ giữ nguyên
# )
# Warmup + Cosine
initial_lr = 0.001
total_steps = 160 * 1279 #here

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=total_steps,
    end_learning_rate=0.00001,
    power=1.0  # Linear decay
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
conformer.add_featurizers(speech_featurizer, text_featurizer)
conformer.compile(optimizer=optimizer)
for layer in conformer.layers:
    print(f"Layer {layer.name}:")
    print(f"  Trainable: {layer.trainable}")
    print(f"  Total Parameters: {layer.count_params()}")
    print(f"  Trainable Parameters: {sum([K.count_params(p) for p in layer.trainable_weights])}")
    print(f"  Non-trainable Parameters: {sum([K.count_params(p) for p in layer.non_trainable_weights])}")
batch_size = args.bs or config.learning_config.running_config.batch_size

#Keys in batch[0]: dict_keys(['inputs', 'inputs_length', 'predictions', 'predictions_length'])
#Keys in batch[1]: dict_keys(['labels', 'labels_length'])

blank_id = text_featurizer.blank  #(0)

beam_decoded = []
wer_scores = []
beam_decoded_test = []
wer_scores_test = []

train_losses = []
valid_losses = []
epochs_recorded = []


EPOCHS = 300
start_epoch = 0
# =============================================================================
# Training Data        
#Logits shape: (1, 925, 90) (batch_size, time_steps, num_classes)
        
        # probs. Giả sử batch với 2 mẫu, mỗi mẫu có 3 bước thời gian và mỗi bước thời gian có 4 lớp 
        # [[[0.191, 0.640, 0.021, 0.148], 
        # [0.213, 0.577, 0.206, 0.004], 
        # [0.270, 0.408, 0.200, 0.122]],

        # [[0.406, 0.276, 0.142, 0.176], 
        # [0.061, 0.706, 0.197, 0.036], 
        # [0.232, 0.375, 0.392, 0.001]]]
        
# =============================================================================
# Đường dẫn thư mục lưu checkpoint
checkpoint_dir = './checkpoints7' #here
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
best_ckpt_path = os.path.join(checkpoint_dir, 'best_val_loss')
best_val_loss_path = os.path.join(checkpoint_dir, 'best_val_loss.txt')
# Đối tượng checkpoint lưu optimizer, model và epoch
ckpt = tf.train.Checkpoint(optimizer=conformer.optimizer, model=conformer, epoch=tf.Variable(start_epoch))
# Quản lý checkpoint (giữ tối đa 3 file)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

# Khôi phục checkpoint nếu có
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print(f"Restored epoch: {ckpt.epoch.numpy()}")
    step = conformer.optimizer.iterations.numpy()
    lr = conformer.optimizer.learning_rate(step).numpy()
    print(f"Restored learning rate: {lr:.6f} at step {step}")
    print(f"✅ Restored from last checkpoint at epoch {int(ckpt.epoch.numpy())}")
    start_epoch = int(ckpt.epoch.numpy())

    # Load best validation loss nếu file tồn tại
    if os.path.exists(best_val_loss_path):
        try:
            with open(best_val_loss_path, 'r') as f:
                best_val_loss = float(f.read().strip())
            print(f"📉 Current best Validation Loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"⚠️ Could not read best_val_loss.txt: {e}")
            best_val_loss = float('inf')
    else:
        print("ℹ️ No best_val_loss.txt found, starting with inf.")
        best_val_loss = float('inf')
else:
    print("🔄 Initializing from scratch.")
    start_epoch = 0
    best_val_loss = float('inf')
train_data_loader = train_dataset.create(batch_size)
valid_data_loader = valid_dataset.create(batch_size)
test_data_loader = test_dataset.create(batch_size)
import ast

metrics_path = "loss_plots7/log_metrics.txt" #here
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        lines = f.readlines()
        if len(lines) >= 4:
            valid_losses = ast.literal_eval(lines[0])
            train_losses = ast.literal_eval(lines[1])
            wer_scores_test = ast.literal_eval(lines[2])
            wer_scores = ast.literal_eval(lines[3])
        else:
            valid_losses, train_losses, wer_scores_test, wer_scores = [], [], [], []
else:
    valid_losses, train_losses, wer_scores_test, wer_scores = [], [], [], []
#with tf.device('/GPU:0')
for epoch in range(start_epoch, EPOCHS):
    #print(f"Length of train_dataset {len(train_data_loader)}")
    pred_decoded = []
    true_decoded = []
    pred_decoded_test = []
    true_decoded_test = []
    totlen = len(train_data_loader)
    totloss = 0
    nu_train_batch = 0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for k, batch in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
        #{'inputs': <tf.Tensor: shape=(8, 3700, 80, 1), 8 là batch_size
        metrics = conformer.train_step(batch)  #Step 1

        totloss += abs(metrics['loss'].numpy())
        nu_train_batch += 1
        del batch, metrics
        

    totloss = totloss/(nu_train_batch+0.0005)

# =============================================================================
# Validating Data        
# =============================================================================
    
    #print(len(valid_data_loader))
    lossval = 0
    n_samples = 0
    wer_total = 0
    wer_count = 0

    for i, batch in tqdm(enumerate(valid_data_loader), total= len(valid_data_loader)):
        
        labels, labels_len = batch[1]['labels'], batch[1]['labels_length']
        metrics = conformer.test_step(batch)
        
        # Tính loss valid
        lossval += abs(metrics['loss'].numpy())
        n_samples += 1

        # ==== Thêm phần tính WER ======
        y_preds = metrics["y_pred"]
        logits, logits_len = y_preds['logits'], y_preds['logits_length']
        probs = softmax(logits)

        for i, (p, l, label, ll) in enumerate(zip(probs, logits_len, labels, labels_len)):
            # p: length x characters
            pred = p[:l].argmax(-1) #Lấy chỉ số class có xác suất cao nhất.
            decoded_prediction = []
            previous = blank_id

            # remove the repeting characters and the blanck characters
            #[b, b, l, l, blank, l, e, e, blank, blank] → b, l, e.
            for p in pred:
                if (p != previous or previous == blank_id) and p != blank_id:
                    decoded_prediction.append(p)
                previous = p

            if len(decoded_prediction) == 0:
                decoded = ""
            else:
                decoded = text_featurizer.iextract([decoded_prediction]).numpy()[0].decode('utf-8')
            pred_decoded.append(decoded)
            label_len = tf.math.reduce_sum(tf.cast(label != 0, tf.int32))
            true_decoded.append(text_featurizer.iextract([label[:label_len]]).numpy()[0].decode('utf-8'))


#=====================================================
# Test Data
#=====================================================
    #print(len(valid_data_loader))
    n_samples_test = 0
    wer_total_test = 0
    wer_count_test = 0


    for i, batch in tqdm(enumerate(test_data_loader), total= len(test_data_loader)):
        
        labels, labels_len = batch[1]['labels'], batch[1]['labels_length']
        metrics = conformer.test_step(batch)
        
        n_samples_test += 1

        # ==== Thêm phần tính WER ======
        y_preds = metrics["y_pred"]
        logits, logits_len = y_preds['logits'], y_preds['logits_length']
        probs = softmax(logits)

        #beam[i] là kết quả dự đoán cho sample thứ i trong batch.
        for i, (p, l, label, ll) in enumerate(zip(probs, logits_len, labels, labels_len)):
            # p: length x characters
            pred = p[:l].argmax(-1) #Lấy chỉ số class có xác suất cao nhất.
            decoded_prediction = []
            previous = blank_id

            # remove the repeting characters and the blanck characters
            #[b, b, l, l, blank, l, e, e, blank, blank] → b, l, e.
            for p in pred:
                if (p != previous or previous == blank_id) and p != blank_id:
                    decoded_prediction.append(p)
                previous = p

            if len(decoded_prediction) == 0:
                decoded = ""
            else:
                decoded = text_featurizer.iextract([decoded_prediction]).numpy()[0].decode('utf-8')
            pred_decoded_test.append(decoded)
            label_len = tf.math.reduce_sum(tf.cast(label != 0, tf.int32))
            true_decoded_test.append(text_featurizer.iextract([label[:label_len]]).numpy()[0].decode('utf-8'))

    if all(len(ref.strip().split()) == 0 for ref in true_decoded_test):
        print("WARNING: All reference sentences are empty! Cannot compute WER.")
        wer_test = None  # hoặc wer = 0 tùy ý
    else:
        wer_test = wer_metric.compute(predictions=pred_decoded_test, references=true_decoded_test)
    if all(len(ref.strip()) == 0 for ref in true_decoded):
        continue  # bỏ qua batch này

    if all(len(ref.strip().split()) == 0 for ref in true_decoded):
        print("WARNING: All reference sentences are empty! Cannot compute WER.")
        wer = None  # hoặc wer = 0 tùy ý
    else:
        wer = wer_metric.compute(predictions=pred_decoded, references=true_decoded)

    lossval = lossval / n_samples
    wer_scores.append(wer)
    wer_scores_test.append(wer_test)
    print(f'average loss valid = {lossval} | WER_VALID = {wer:.4f}| WER_TEST = {wer_test:.4f}')

    #del valid_data_loader

    #gc.collect()  # Dọn rác bộ nhớ CPU (tốt khi dùng với NumPy, TF object cũ)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {totloss:.4f} | Valid Loss: {lossval:.4f}")
    train_losses.append(totloss)
    valid_losses.append(lossval)
    epochs_recorded.append(epoch + 1)  # Epoch hiện tại
    
    # Cập nhật số epoch hiện tại
    ckpt.epoch.assign(epoch + 1)
    # Lưu checkpoint
    save_path = ckpt_manager.save()

    # Lưu checkpoint tốt nhất nếu lossval giảm
    if lossval < best_val_loss:
        best_val_loss = lossval

        ckpt.write(best_ckpt_path)

        # Lưu loss vào file (tuỳ chọn)
        with open(os.path.join(checkpoint_dir, 'best_val_loss.txt'), 'w') as f:
            f.write(str(best_val_loss))

        print(f"🏅 Saved new BEST checkpoint with VAL LOSS {lossval:.4f} at {best_ckpt_path}")

        # In learning rate hiện tại (nếu cần)
        step = conformer.optimizer.iterations.numpy()
        lr = conformer.optimizer.learning_rate(step).numpy()
        print(f"🔁 Current learning rate at step {step}: {lr:.6f}")
    # Sau mỗi 5 epoch:
    epochs = list(range(1, len(train_losses) + 1))
    if (epoch + 1) % 5 == 0:
        plt.figure(figsize=(16, 8))
        plt.plot(epochs, train_losses, label='Train Loss', marker='o')
        plt.plot(epochs, valid_losses, label='Valid Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Lưu hình với tên theo epoch
        save_path = f"loss_plots7/loss_epoch_{epoch+1}.png" #here
        plt.savefig("loss_plots7/loss_plot_latest.png")

        # Vẽ biểu đồ cho WER
        plt.figure(figsize=(16, 8))
        plt.plot(epochs, wer_scores, label='Valid WER', color='red', marker='o')
        plt.plot(epochs, wer_scores_test, label='Test WER', color='blue', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('WER')
        plt.title('Validation and Test WER per Epoch')
        plt.legend()
        plt.grid(True)

        # Lưu hình biểu đồ WER
        save_path_wer = f"loss_plots7/wer_epoch_{epoch+1}.png"
        plt.savefig("loss_plots7/wer_plot_latest.png") #here
    # Ghi lại các danh sách metrics theo từng dòng
    with open("loss_plots7/log_metrics.txt", "w") as f: #here
        f.write(f"{valid_losses}\n")     # Dòng 1: Valid Loss
        f.write(f"{train_losses}\n")     # Dòng 2: Train Loss
        f.write(f"{wer_scores_test}\n")  # Dòng 3: Test WER
        f.write(f"{wer_scores}\n")       # Dòng 4: Valid WER
