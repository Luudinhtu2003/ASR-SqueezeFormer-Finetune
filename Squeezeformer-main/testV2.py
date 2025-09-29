# Đánh giá lại WER trên toàn bộ tập train hoặc test, dựa vào file tsv
import os
from tqdm import tqdm
import argparse
from scipy.special import softmax
from datasets import load_metric
import tensorflow as tf
from tensorflow.keras import backend as K

from src.configs.config import Config
from src.datasets.asr_dataset import ASRSliceDataset
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.utils import env_util, file_util

wer_metric = load_metric("wer")  # chỉ cần load 1 lần
logger = env_util.setup_environment()

physical_devices = tf.config.list_physical_devices('GPU') 

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

os.makedirs("loss_plots1", exist_ok=True)
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
    parser.add_argument("--input_padding", type=int, default=801)
    parser.add_argument("--label_padding", type=int, default=132)

    # Architecture arguments
    parser.add_argument("--fixed_arch", default=None, help="force fixed architecture")

    # Decoding arguments
    parser.add_argument("--beam_size", type=int, default=None, help="ctc beam size")

    args = parser.parse_args()
    return args


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
print("Labels không có blank:")
print(text_featurizer.non_blank_tokens)
print("Length của labels không có blank:", len(text_featurizer.non_blank_tokens))
for i, token in enumerate(text_featurizer.non_blank_tokens):
    print(f"{i+1}: '{token}'")

tf.random.set_seed(0)

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

dataset_val_path = r"F:\Luu_Dinh_Tu\Project_2\DATN\datasets\test_processed.tsv"

if dataset_val_path is not None:
    logger.info(f"dataset valid: at {dataset_val_path}")
    config.learning_config.eval_dataset_config.data_paths = [dataset_val_path]
else:
    raise ValueError("specify the manifest file path using --dataset_valid_path")

valid_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.eval_dataset_config) #Load parameters for valid
)

def build_conformer(config, vocab_size, featurizer_shape):
    model = ConformerCtc(
        **config.model_config, 
        vocabulary_size=vocab_size, 
    )
    model.make(featurizer_shape)
    return model

conformer = build_conformer(config, text_featurizer.num_classes, speech_featurizer.shape)
temp_model = build_conformer(config, 128, speech_featurizer.shape)


#conformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2))
initial_lr = 0.001
total_steps = 100 * 644

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=total_steps,
    end_learning_rate=0.00001,
    power=1.0  # Linear decay
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

conformer.add_featurizers(speech_featurizer, text_featurizer)
conformer.compile(optimizer=optimizer)
batch_size = args.bs or config.learning_config.running_config.batch_size

#Keys in batch[0]: dict_keys(['inputs', 'inputs_length', 'predictions', 'predictions_length'])
#Keys in batch[1]: dict_keys(['labels', 'labels_length'])

blank_id = text_featurizer.blank  #(0)

true_decoded = []
pred_decoded = []
beam_decoded = []
train_losses = []
valid_losses = []
epochs_recorded = []
wer_scores = []

EPOCHS = 1
start_epoch = 0

# Đường dẫn thư mục lưu checkpoint
checkpoint_dir = './checkpoints5'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
best_ckpt_path = os.path.join(checkpoint_dir, 'best_wer')
best_wer = float('inf')  # Khởi đầu với giá trị WER cao
# Đối tượng checkpoint lưu optimizer, model và epoch
ckpt = tf.train.Checkpoint(optimizer=conformer.optimizer, model=conformer, epoch=tf.Variable(start_epoch))
# Quản lý checkpoint (giữ tối đa 5 file)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

#train_function = model.make_train_function()
# Khôi phục checkpoint nếu có
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    # Khôi phục checkpoint có WER thấp nhất (tốt nhất)
    # ckpt.restore(best_ckpt_path)
    # print("✅ Restored BEST WER checkpoint.")

    print(f"✅ Restored from last checkpoint at epoch {int(ckpt.epoch.numpy())}")
    start_epoch = int(ckpt.epoch.numpy())
    # Load best_wer nếu có file lưu
    best_wer_path = os.path.join(checkpoint_dir, 'best_wer.txt')
    if os.path.exists(best_wer_path):
        with open(best_wer_path, 'r') as f:
            best_wer = float(f.read().strip())
            print(f"📉 Current best WER: {best_wer:.4f}")
else:
    print("🔄 Initializing from scratch.")
valid_data_loader = valid_dataset.create(batch_size)


lossval = 0
n_samples = 0
wer_total = 0
wer_count = 0
#classes = text_featurizer.tokens
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

    if args.beam_size is not None:
        beam = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits, perm=[1, 0, 2]), logits_len, beam_width=args.beam_size, top_paths=1,
        )
        # Chuyển về (Time, batch, classes)
        beam = tf.sparse.to_dense(beam[0][0]).numpy()
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
        pred_decoded.append(decoded)
        label_len = tf.math.reduce_sum(tf.cast(label != 0, tf.int32))
        true_decoded.append(text_featurizer.iextract([label[:label_len]]).numpy()[0].decode('utf-8'))

        #Use beam search
        if args.beam_size is not None:
            b = beam[i]
            previous = blank_id

            # remove the repeting characters and the blanck characters
            beam_prediction = []
            for p in b:
                if (p != previous or previous == blank_id) and p != blank_id:
                    beam_prediction.append(p)
                previous = p

            if len(beam_prediction) == 0:
                decoded = ""
            else:
                decoded = text_featurizer.iextract([beam_prediction]).numpy()[0].decode('utf-8')
            beam_decoded.append(decoded)
            #del batch, metrics

if all(len(ref.strip().split()) == 0 for ref in true_decoded):
    print("WARNING: All reference sentences are empty! Cannot compute WER.")
    wer = None  # hoặc wer = 0 tùy ý
else:
    wer = wer_metric.compute(predictions=pred_decoded, references=true_decoded)

lossval = lossval / n_samples
wer_scores.append(wer)
print(f"Number of predictions: {len(pred_decoded)}")
print(f"Number of references: {len(true_decoded)}")
print(f"number of wers_scores: {len(wer_scores)}")
average_wer = sum(wer_scores) / len(wer_scores)
print(f'average loss valid = {lossval} | WER = {average_wer:.4f}')






        