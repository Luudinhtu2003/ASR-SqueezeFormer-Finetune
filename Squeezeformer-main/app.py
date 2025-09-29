"""
Check microphone recording and display mel spectrogram.
"""


import pyaudio
import numpy as np
import keyboard
import time
import os
import argparse
from scipy.special import softmax
import tensorflow as tf
import time

from src.configs.config import Config
from src.datasets.asr_dataset import ASRSliceDataset
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.utils import env_util
import wave

def local_agreement(prev, curr, n=2, window=10):
    """
    So sánh phần cuối prev với đầu curr, tìm đoạn trùng dài nhất >= n.
    """
    max_overlap = 0
    agreed = ""
    # Chỉ xét tối đa window ký tự cuối của prev và đầu của curr
    for i in range(1, window+1):
        if prev[-i:] == curr[:i]:
            max_overlap = i
    if max_overlap >= n:
        agreed = curr[:max_overlap]
        remain = curr[max_overlap:]
        return agreed, remain
    else:
        return "", curr
# Tên file để lưu
output_filename = "recorded_audio.wav"
logger = env_util.setup_environment()

physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    parser.add_argument("--input_padding", type=int, default=400)
    parser.add_argument("--label_padding", type=int, default=160)

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

def build_conformer(config, vocab_size, featurizer_shape):
    model = ConformerCtc(
        **config.model_config, 
        vocabulary_size=vocab_size, 
    )
    model.make(featurizer_shape)
    return model

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


conformer = build_conformer(config, 90, speech_featurizer.shape)

#conformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2))
initial_lr = 0.001
total_steps = 300 * 644
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=10000,
    end_learning_rate=0.00001,
    power=1.0  # Linear decay
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

conformer.add_featurizers(speech_featurizer, text_featurizer)
conformer.compile(optimizer=optimizer)
batch_size = args.bs or config.learning_config.running_config.batch_size
blank_id = text_featurizer.blank  #(0)

EPOCHS = 300
start_epoch = 0

checkpoint_dir = './checkpoints11' #here
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
# Bây giờ bạn có thể đưa audio_data vào hàm convert_waveform_to_encoded_wav
inference = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.train_dataset_config) # Load parameters for train
)
# Các tham số của micro
FORMAT = pyaudio.paInt16  # Định dạng âm thanh
CHANNELS = 1  # Mono
RATE = 16000  # Sample rate (16 kHz)
CHUNK = 1024  # Kích thước mỗi chunk
MAX_DURATION = 8  # Thời gian tối đa ghi âm (5 giây)
classes = ['', '', '', ' ', 'n', 'h', 't', 'i', 'c', 'g', 'a', 'm', 'u', 'đ', 'à', 'o', 
           'ư', 'v', 'l', 'r', 'á', 'y', 'b', 'p', 'ô', 'k', 's', 'ó', 'ế', 'ạ', 'ộ', 'ờ',
             'ệ', 'ả', 'ê', 'ì', 'd', 'â', 'ố', 'ớ', 'ấ', 'ơ', 'ề', 'q', 'ủ', 'ể', 'ă', 'ị',
            'ợ', 'í', 'ậ', 'e', 'x', 'ầ', 'ự', 'ú', 'ữ', 'ọ', 'ứ', 'ã', 'ở', 'ồ', 'ụ', 'ắ',
             'ừ', 'ổ', 'ò', 'ũ', 'ù', 'ặ', 'ý', 'ỉ', 'ẽ', 'ỏ', 'ử', 'ằ', 'é', 'ĩ', 'ễ', 'ẩ', 
             'ẫ', 'ỗ', 'ẹ', 'ỹ', 'ẻ', 'ỳ', 'è', 'õ', 'ỡ', 'ẳ']
def _greedy_decode(logits):
        """Decode argmax of logits and squash in CTC fashion."""
        label_dict = {n: c for n, c in enumerate(classes)}
        prev_c = None
        out = []
        for n in tf.argmax(logits, axis=1):
            c = label_dict.get(int(n), "")  # if not in labels, then assume it's ctc blank char
            if c != prev_c:
                out.append(c)
            prev_c = c
        return "".join(out)
# Khởi tạo PyAudio
p = pyaudio.PyAudio()

SEGMENT_DURATION = 2  # giây
SEGMENT_SAMPLES = SEGMENT_DURATION * RATE

print("Nhấn 'r' để bắt đầu ghi âm streaming, nhấn 'esc' để thoát.")

while True:
    if keyboard.is_pressed('esc'):
        print("Exiting...")
        break

    if keyboard.is_pressed('r'):
        print("Streaming recording started...")
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        segment_frames = []
        start_time = time.time()
        decoded_full = ""
        prev_decoded = ""

        while keyboard.is_pressed('r'):
            data = stream.read(CHUNK)
            frame_array = np.frombuffer(data, dtype=np.int16)
            frames.append(frame_array)
            segment_frames.append(frame_array)

            # Nếu đã đủ dữ liệu cho 1 segment
            if sum(len(f) for f in segment_frames) >= SEGMENT_SAMPLES:
                audio_data = np.concatenate(segment_frames, axis=0)
                audio_data = tf.expand_dims(audio_data, axis=-1)

                # Tiền xử lý và nhận dạng
                inputs_tensor, inputs_length_tensor = inference.preprocess_from_mic(audio_data)
                inputs_tensor = tf.expand_dims(inputs_tensor, axis=0)
                inputs_length_tensor = tf.expand_dims(inputs_length_tensor, axis=0)

                inputs = {
                    "inputs": inputs_tensor,
                    "inputs_length": inputs_length_tensor
                }

                outputs = conformer(inputs, training=False)
                logits = outputs['logits']
                curr_decoded = _greedy_decode(logits[0])

                # LocalAgreement n=2
                agreed, remain = local_agreement(prev_decoded, curr_decoded, n=2)
                decoded_full += agreed
                prev_decoded = curr_decoded

                print(f"[Segment] Decoded: {curr_decoded}")
                print(f"[Segment] Agreed: {agreed}")
                print(f"[Segment] Full so far: {decoded_full}")

                segment_frames = []  # reset segment buffer

            if time.time() - start_time >= MAX_DURATION:
                print("Reached max duration.")
                break

        # Xử lý phần còn lại nếu có
        # ...existing code...
        if segment_frames:
            audio_data = np.concatenate(segment_frames, axis=0)
            audio_data = tf.expand_dims(audio_data, axis=-1)
            inputs_tensor, inputs_length_tensor = inference.preprocess_from_mic(audio_data)
            inputs_tensor = tf.expand_dims(inputs_tensor, axis=0)
            inputs_length_tensor = tf.expand_dims(inputs_length_tensor, axis=0)
            inputs = {
                "inputs": inputs_tensor,
                "inputs_length": inputs_length_tensor
            }
            outputs = conformer(inputs, training=False)
            logits = outputs['logits']
            curr_decoded = _greedy_decode(logits[0])
            agreed, remain = local_agreement(prev_decoded, curr_decoded, n=2)
            decoded_full += agreed + remain  # Ghép cả phần còn lại

        stream.stop_stream()
        stream.close()
        print("\n======= Final Output =======")
        print(decoded_full.strip())

    time.sleep(0.05)