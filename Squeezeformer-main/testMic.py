import pyaudio
import numpy as np
import keyboard
import time
import os
from tqdm import tqdm
import argparse
from scipy.special import softmax
import datasets
import gc
import matplotlib.pyplot as plt
import IPython.display as display
from datasets import load_metric
import tensorflow as tf
import time

from src.configs.config import Config
from src.datasets.asr_dataset import ASRSliceDataset
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.utils import env_util, file_util

logger = env_util.setup_environment()

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

os.makedirs("loss_plots", exist_ok=True)
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
    parser.add_argument("--input_padding", type=int, default=500)
    parser.add_argument("--label_padding", type=int, default=530)

    # Architecture arguments
    parser.add_argument("--fixed_arch", default=None, help="force fixed architecture")

    # Decoding arguments
    parser.add_argument("--beam_size", type=int, default=None, help="ctc beam size")

    args = parser.parse_args()
    return args


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
NUM_LAYERS_IN_BLOCK = 4

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})
env_util.setup_devices([args.device], cpu=args.cpu)


speech_featurizer = TFSpeechFeaturizer(config.speech_config)

logger.info("Use SentencePiece ...")
text_featurizer = SentencePieceFeaturizer(config.decoder_config)

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

# Các tham số của micro
FORMAT = pyaudio.paInt16  # Định dạng âm thanh
CHANNELS = 1  # Mono
RATE = 16000  # Sample rate (16 kHz)
CHUNK = 1024  # Kích thước mỗi chunk
MAX_DURATION = 5  # Thời gian tối đa ghi âm (5 giây)

# Khởi tạo PyAudio
p = pyaudio.PyAudio()

# Mở stream ghi âm từ micro
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Press 'r' to start recording, release to stop.")

frames = []  # Dùng để lưu dữ liệu thu âm

# Thời gian bắt đầu
start_time = None

# Bắt đầu thu âm khi nhấn phím "r" và dừng khi thả phím "r" hoặc vượt quá 5 giây
while True:
    if keyboard.is_pressed('r'):
        if start_time is None:
            start_time = time.time()  # Ghi lại thời gian khi bắt đầu thu âm
            print("Recording...")

        # Ghi âm vào frames
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

        # Kiểm tra thời gian đã thu âm
        if time.time() - start_time >= MAX_DURATION:
            print("Reached maximum duration (5 seconds). Stopping recording.")
            break

    elif start_time is not None:
        print("Recording stopped.")
        break

# Đóng stream sau khi thu âm xong
stream.stop_stream()
stream.close()
p.terminate()

# Chuyển đổi toàn bộ frames thành một numpy array
audio_data = np.concatenate(frames, axis=0)

audio_data = tf.expand_dims(audio_data, axis=-1) 
# Kiểm tra kiểu dữ liệu và shape của audio_data
print(f"Audio data shape: {audio_data.shape}")
print(f"Audio data dtype: {audio_data.dtype}")

def build_conformer(config, vocab_size, featurizer_shape):
    model = ConformerCtc(
        **config.model_config, 
        vocabulary_size=vocab_size, 
    )
    model.make(featurizer_shape)
    return model
conformer = build_conformer(config, 128, speech_featurizer.shape)
conformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
# Load weights
if args.saved:
    conformer.load_weights(args.saved, by_name=True)
    logger.info(f"Weights loaded from {args.saved}")
else:
    logger.warning("Failed.")
# Kiểm tra lại trạng thái của các lớp
for layer in conformer.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

conformer.add_featurizers(speech_featurizer, text_featurizer)
# Bây giờ bạn có thể đưa audio_data vào hàm convert_waveform_to_encoded_wav
inference = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.train_dataset_config) # Load parameters for train
)
batch_size = args.bs or config.learning_config.running_config.batch_size
blank_id = text_featurizer.blank  #(0)


# Tiền xử lý âm thanh từ mic
inputs_tensor, inputs_length_tensor = inference.preprocess_from_mic(audio_data)

# Thêm một chiều để làm batch size (giả sử batch size = 1)
inputs_tensor = tf.expand_dims(inputs_tensor, axis=0)  # shape sẽ thành (1, T, F, 1)
inputs_length_tensor = tf.expand_dims(inputs_length_tensor, axis=0)
# Tạo dictionary inputs cho mô hình
inputs = {
    "inputs": inputs_tensor,  # Tensor chứa đặc trưng âm thanh
    "inputs_length": inputs_length_tensor  # Tensor chứa độ dài chuỗi
}
true_decoded = []
pred_decoded = []
beam_decoded = []
start_time = time.time()
outputs = conformer(inputs, training=False)
# for key, value in outputs.items():
#     print(f"Key: '{key}'")
#     print(f"  + Shape: {value.shape if hasattr(value, 'shape') else 'Không có shape'}")
#     print(f"  + Value:\n{value}\n")

logits, logits_len = outputs['logits'], outputs['logits_length']
probs = softmax(logits)

if args.beam_size is not None:
        beam = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits, perm=[1, 0, 2]), logits_len, beam_width=args.beam_size, top_paths=1,
        )
        # Chuyển về (Time, batch, classes)
        beam = tf.sparse.to_dense(beam[0][0]).numpy()
#beam[i] là kết quả dự đoán cho sample thứ i trong batch.
for i, (p, l) in enumerate(zip(probs, logits_len)):
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
    end_time1 = time.time()  
    print(f"Predicted: {decoded}")
    print(f"[Greedy] Predicted: {decoded}")
    print(f"[Greedy] Time: {end_time1 - start_time:.4f} seconds")
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
        end_time2 = time.time()  
        print(f"Beam decoded: {decoded}")
        print(f"[Beam] Beam decoded: {decoded}")
        print(f"[Beam] Time: {end_time2 - end_time1:.4f} seconds")
