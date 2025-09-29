"""
Check if freezing layers in the encoder works correctly
"""
import os
import argparse
from datasets import load_metric
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from src.configs.config import Config
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.utils import env_util

# Kiểm tra việc đóng băng có thành công không
wer_metric = load_metric("wer")  # chỉ cần load 1 lần
logger = env_util.setup_environment()

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
encoder_layer = conformer.get_layer("conformer_encoder")

# Lặp qua tất cả các layers bên trong encoder
for i, layer in enumerate(encoder_layer.layers):
    if layer.name in ["conformer_encoder_block_15", "dense"]:
        layer.trainable = True  # Unfreeze layer 14 và 15
    else:
        layer.trainable = False  # Freeze các layer còn lại

# Kiểm tra lại trạng thái trainable của từng layer
for i, layer in enumerate(encoder_layer.layers):
    print(f"Layer {layer.name}: Trainable: {layer.trainable}")
conformer.decoder.trainable = True
# Ví dụ giả định block được lưu dạng list: encoder.blocks
# def set_trainable_recursively(layer):
#     layer.trainable = True
#     for sublayer in getattr(layer, 'layers', []):
#         set_trainable_recursively(sublayer)

# # Unfreeze block 14 và 15 đệ quy toàn bộ
# for i in range(14, 16):
#     block = conformer.encoder.get_layer(f"conformer_encoder_block_{i}")
#     set_trainable_recursively(block)
#     print(f"Unfrozen recursively: {block.name}")

conformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3))
conformer.summary(line_length=150)

# In ra thông tin chi tiết từng layer bên trong encoder
for i, layer in enumerate(conformer.encoder.layers):
    print(f"Layer {i}: {layer.name}")
    # print(f"  Trainable: {layer.trainable}")
    # print(f"  Total Parameters: {layer.count_params()}")
    # print(f"  Trainable Parameters: {sum([K.count_params(p) for p in layer.trainable_weights])}")
    # print(f"  Non-trainable Parameters: {sum([K.count_params(p) for p in layer.non_trainable_weights])}")
    # print("-" * 50)
for layer in conformer.layers:
    print(f"Layer {layer.name}:")
    print(f"  Trainable: {layer.trainable}")
    print(f"  Total Parameters: {layer.count_params()}")
    print(f"  Trainable Parameters: {sum([K.count_params(p) for p in layer.trainable_weights])}")
    print(f"  Non-trainable Parameters: {sum([K.count_params(p) for p in layer.non_trainable_weights])}")


