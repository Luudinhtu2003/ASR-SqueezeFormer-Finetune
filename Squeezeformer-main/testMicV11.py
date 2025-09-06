import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import pyaudio
import numpy as np
import keyboard
import time
import argparse
from scipy.special import softmax
import time

from src.configs.config import Config
from src.datasets.asr_dataset import ASRSliceDataset
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.utils import env_util

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
    parser.add_argument("--input_padding", type=int, default=801)
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

def build_conformer(config, vocab_size, featurizer_shape):
    model = ConformerCtc(
        **config.model_config, 
        vocabulary_size=vocab_size, 
    )
    model.make(featurizer_shape)
    return model
conformer = build_conformer(config, 93, speech_featurizer.shape)

#conformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2))
initial_lr = 0.001
total_steps = 300 * 644
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=10000,
    end_learning_rate=0.00001,
    power=1.0  # Linear 
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

conformer.add_featurizers(speech_featurizer, text_featurizer)
conformer.compile(optimizer=optimizer)

EPOCHS = 300
start_epoch = 0

checkpoint_dir = './checkpoints7' #here
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

# Các tham số của micro
import threading
import scipy.io.wavfile
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 256
MAX_DURATION = 8  # tổng thời gian ghi âm tối đa (giây)
PROCESS_INTERVAL = 1.75  # mỗi 2s xử lý 1 lần
recording = True
latencies = []
rtfs = []
while True:
    audio_buffer = np.array([], dtype=np.int16)
    recording = True
    audio_chunks = []
    buffer_lock = threading.Lock()
    def record_audio():
        global recording
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Nhấn 'r' để bắt đầu ghi âm, nhấn lại 'r' để dừng.")
        while not keyboard.is_pressed('r'):
            time.sleep(0.01)
        print("Đang ghi âm...")
        start_time = time.time()
        while keyboard.is_pressed('r'):
            data = stream.read(CHUNK)
            chunk_data = np.frombuffer(data, dtype=np.int16)
            with buffer_lock:
                audio_chunks.append(chunk_data)
            if (time.time() - start_time) > MAX_DURATION:
                break
        #print("Dừng ghi âm.")
        recording = False
        stream.stop_stream()
        stream.close()
        p.terminate()
        # Lưu file WAV sau khi ghi xong
        with buffer_lock:
            audio_buffer = np.concatenate(audio_chunks, axis=0)
        scipy.io.wavfile.write("recorded_audio.wav", RATE, audio_buffer.astype(np.int16))

    def recognize_audio():
        global recording
        committed_text = ""
        prev_text = ""
        last_processed_samples = 0
        latency_list = []
        rtf_list = []

        while True:
            with buffer_lock:
                audio_buffer = np.concatenate(audio_chunks, axis=0) if audio_chunks else np.array([], dtype=np.int16)
            current_samples = len(audio_buffer)

            if recording:
                if current_samples - last_processed_samples >= PROCESS_INTERVAL * RATE:
                    # Lấy buffer audio để nhận dạng
                    if current_samples < MAX_DURATION * RATE:
                        buf = audio_buffer
                    else:
                        buf = audio_buffer[-MAX_DURATION * RATE:]

                    audio_capture_time = time.time()  # Thời điểm audio được lấy để xử lý

                    # --- BẮT ĐẦU PHẦN NHẬN DIỆN ---
                    start_proc_time = time.time()

                    audio_data = tf.expand_dims(buf, axis=-1)
                    inputs_tensor, inputs_length_tensor = inference.preprocess_from_mic(audio_data)
                    inputs_tensor = tf.expand_dims(inputs_tensor, axis=0)
                    inputs_length_tensor = tf.expand_dims(inputs_length_tensor, axis=0)
                    inputs = {
                        "inputs": inputs_tensor,
                        "inputs_length": inputs_length_tensor
                    }

                    outputs = conformer(inputs, training=False)
                    logits, logits_len = outputs['logits'], outputs['logits_length']
                    probs = softmax(logits)
                    blank_id = text_featurizer.blank
                    pred = probs[0][:logits_len[0]].argmax(-1)
                    decoded_prediction = []
                    previous = blank_id
                    for p_ in pred:
                        if (p_ != previous or previous == blank_id) and p_ != blank_id:
                            decoded_prediction.append(p_)
                        previous = p_
                    if len(decoded_prediction) == 0:
                        decoded = ""
                    else:
                        decoded = text_featurizer.iextract([decoded_prediction]).numpy()[0].decode('utf-8')

                    end_proc_time = time.time()
                    # --- KẾT THÚC PHẦN NHẬN DIỆN ---

                    # Tính thời gian xử lý và RTF
                    processing_time = end_proc_time - start_proc_time
                    audio_duration = len(buf) / RATE
                    rtf = processing_time / audio_duration
                    rtf_list.append(rtf)

                    # Tính latency: thời gian từ lúc audio được lấy đến lúc kết quả có
                    latency = end_proc_time - audio_capture_time
                    latency_list.append(latency)

                    # Commit phần giống nhau và in ra
                    if decoded.startswith(committed_text):
                        new_part = decoded[len(committed_text):]
                        if new_part:
                            print(new_part, end='', flush=True)
                            committed_text = decoded
                    prev_text = decoded
                    last_processed_samples = current_samples

                else:
                    time.sleep(0.05)
            else:
                # Xử lý phần còn lại khi dừng ghi âm (tương tự trên)
                if last_processed_samples < current_samples:
                    if current_samples < MAX_DURATION * RATE:
                        buf = audio_buffer
                    else:
                        buf = audio_buffer[-MAX_DURATION * RATE:]

                    audio_capture_time = time.time()

                    start_proc_time = time.time()
                    audio_data = tf.expand_dims(buf, axis=-1)
                    inputs_tensor, inputs_length_tensor = inference.preprocess_from_mic(audio_data)
                    inputs_tensor = tf.expand_dims(inputs_tensor, axis=0)
                    inputs_length_tensor = tf.expand_dims(inputs_length_tensor, axis=0)
                    inputs = {
                        "inputs": inputs_tensor,
                        "inputs_length": inputs_length_tensor
                    }
                    outputs = conformer(inputs, training=False)
                    logits, logits_len = outputs['logits'], outputs['logits_length']
                    probs = softmax(logits)
                    blank_id = text_featurizer.blank
                    pred = probs[0][:logits_len[0]].argmax(-1)
                    decoded_prediction = []
                    previous = blank_id
                    for p_ in pred:
                        if (p_ != previous or previous == blank_id) and p_ != blank_id:
                            decoded_prediction.append(p_)
                        previous = p_
                    if len(decoded_prediction) == 0:
                        decoded = ""
                    else:
                        decoded = text_featurizer.iextract([decoded_prediction]).numpy()[0].decode('utf-8')

                    end_proc_time = time.time()

                    processing_time = end_proc_time - start_proc_time
                    audio_duration = len(buf) / RATE
                    rtf = processing_time / audio_duration
                    rtf_list.append(rtf)

                    latency = end_proc_time - audio_capture_time
                    latency_list.append(latency)

                    if decoded.startswith(committed_text):
                        new_part = decoded[len(committed_text):]
                        if new_part:
                            print(new_part, end='', flush=True)
                            committed_text = decoded
                    prev_text = decoded
                    last_processed_samples = current_samples

                break

        # In phần còn lại chưa commit
        final_part = prev_text[len(committed_text):]
        if final_part:
            print(final_part, end='', flush=True)
        print()

        # In kết quả Latency và RTF trung bình
        if latency_list:
            avg_latency = sum(latency_list) / len(latency_list)
            print(f"\nLatency trung bình: {avg_latency:.3f} giây")
        if rtf_list:
            avg_rtf = sum(rtf_list) / len(rtf_list)
            print(f"RTF trung bình: {avg_rtf:.3f}")
    audio_chunks = []
    recording = True
    t1 = threading.Thread(target=record_audio)
    t2 = threading.Thread(target=recognize_audio)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("Kết thúc.")
