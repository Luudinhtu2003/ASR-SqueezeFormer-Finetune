import streamlit as st
import numpy as np
import tensorflow as tf
from scipy.special import softmax
import soundfile as sf
import os

from src.configs.config import Config
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.datasets.asr_dataset import ASRSliceDataset

config = Config(r"F:\Luu_Dinh_Tu\Project_2\Squeezeformer-main\examples\squeezeformer\configs\squeezeformer-XS.yml")
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

NUM_BLOCKS = config.model_config['encoder_num_blocks']
NUM_LAYERS_IN_BLOCK = 4

speech_featurizer = TFSpeechFeaturizer(config.speech_config)

text_featurizer = SentencePieceFeaturizer(config.decoder_config)

tf.random.set_seed(0)
fixed_arch = 24
# Parse fixed architecture
if fixed_arch is not None:
    fixed_arch = parse_fixed_arch_from_yaml(config.model_config['encoder_fixed_arch'])

    if len(fixed_arch) != NUM_BLOCKS:
        config.model_config['encoder_num_blocks'] = len(fixed_arch)
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
    input_padding_length=800,
    label_padding_length=200,
    **vars(config.learning_config.train_dataset_config) # Load parameters for train
)
batch_size = 32 or config.learning_config.running_config.batch_size
blank_id = text_featurizer.blank  #(0)

st.title("Demo Nhận diện tiếng nói (ASR) với Squeezeformer")

audio_file = st.file_uploader("Tải lên file WAV", type=["wav"])
if audio_file is not None:
    # Đọc file WAV, đảm bảo mono, 16kHz
    audio_np, samplerate = sf.read(audio_file, dtype='int16')
    if len(audio_np.shape) > 1:
        audio_np = audio_np[:, 0]  # chỉ lấy 1 kênh nếu stereo
    if samplerate != 16000:
        st.error("Vui lòng upload file WAV 16kHz!")
    else:
        st.audio(audio_file, format="audio/wav")
        RATE = 16000
        PROCESS_INTERVAL = 2  # giây
        MAX_DURATION = 8      # giây (giống pipeline micro)
        committed_text = ""
        prev_text = ""
        result = ""
        # Chia audio thành các đoạn nhỏ để nhận diện từng phần (commit)
        for end in range(PROCESS_INTERVAL * RATE, len(audio_np) + PROCESS_INTERVAL * RATE, PROCESS_INTERVAL * RATE):
            buf = audio_np[:min(end, len(audio_np))]
            # Giống pipeline micro: expand_dims, dtype int16
            audio_data = tf.expand_dims(buf, axis=-1)
            # Tiền xử lý như pipeline micro
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
            # Commit phần mới
            if decoded.startswith(committed_text):
                new_part = decoded[len(committed_text):]
                if new_part:
                    result += new_part
                    committed_text = decoded
            prev_text = decoded
        st.markdown(f"**Kết quả nhận diện:** {result}")