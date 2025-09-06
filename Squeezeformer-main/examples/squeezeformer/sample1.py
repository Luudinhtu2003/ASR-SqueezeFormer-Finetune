import os
from tqdm import tqdm

#Thư viện xử lý đối số command-line.
import argparse
from scipy.special import softmax # type: ignore
import datasets
import time

import tensorflow as tf # type: ignore

from src.configs.config import Config
from src.datasets.asr_dataset import ASRSliceDataset
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.utils import env_util, file_util


logger = env_util.setup_environment()

#Liệt kê các thiết bị GPU mà TensorFlow có thể nhận diện được
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)

#Bật chế độ tăng dần bộ nhớ, Tf chỉ dùng VRAM cần thiết
# và mở rộng khi cần thiết thay vì chiếm toàn bộ VRAM của GPU
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# __file__ đại diện cho đường dẫn file Python hiện tại, dirname là lấy thư mục cha của nó, 
# tạo đường dẫn tuyệt đối đến file config nằm cùng file Python hiện tại
DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

# Giai phóng GPU, CPU khi train nhiều model
tf.keras.backend.clear_session()

# Checks for the availability of the GPU
device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
    device_name = '/device:CPU:0'
def parse_arguments():
    parser = argparse.ArgumentParser(prog="Conformer Testing")

    #Đường dẫn đến file cấu hình YAML chứa các tham số của mô hình.
    parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
    
    #Sử dụng float16 để tăng tốc và giảm bộ nhớ
    parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")
    
    #ID của GPU để sử dụng. Mặc định là 0.
    parser.add_argument("--device", type=int, default=0, help="Device's id to run test on")
    
    #ép dùng CPU thay vì GPU
    parser.add_argument("--cpu", default=False, action="store_true", help="Whether to only use cpu")
    
    # Đường dẫn đến file checkpoint của model đã được lưu trước đó.
    parser.add_argument("--saved", type=str, default=None, help="Path to saved model")
    
    #Đường dẫn để lưu kết quả sau khi train
    parser.add_argument("--output", type=str, default=None, help="Result filepath")

    # Dataset arguments
    parser.add_argument("--bs", type=int, default=None, help="Test batch size")
    
    #Đường dẫn đến các file .tsv mô tả dữ liệu
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the tsv manifest files")

    parser.add_argument("--input_padding", type=int, default=3700)
    parser.add_argument("--label_padding", type=int, default=530)

    # Architecture arguments
    parser.add_argument("--fixed_arch", default=None, help="force fixed architecture")

    # Decoding arguments, kích thước beam khi decode bằng CTC Beam Search. Gía trị càng cao độ chính xác tăng, speed giảm.
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

args = parse_arguments()
config = Config(args.config)

NUM_BLOCKS = config.model_config['encoder_num_blocks']
NUM_LAYERS_IN_BLOCK = 4

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})
env_util.setup_devices([args.device], cpu=args.cpu)
#--device GPU:0 --cpu False

speech_featurizer = TFSpeechFeaturizer(config.speech_config)

logger.info("Use SentencePiece ...")
text_featurizer = SentencePieceFeaturizer(config.decoder_config)


#bảo đảm yếu tố ngẫu nhiên trong tf
tf.random.set_seed(0)

# Parse fixed architecture
if args.fixed_arch is not None:
    fixed_arch = parse_fixed_arch(args)
    if len(fixed_arch) != NUM_BLOCKS:
        logger.warn(
            f"encoder_num_blocks={config.model_config['encoder_num_blocks']} is " \
            f"different from len(fixed_arch) = {len(fixed_arch)}." \
        )
        logger.warn(f"Changing `encoder_num_blocks` to {len(fixed_arch)}")
        config.model_config['encoder_num_blocks'] = len(fixed_arch)
    logger.info(f"Changing fixed arch: {fixed_arch}")
    config.model_config['encoder_fixed_arch'] = fixed_arch

#Lấy dataset là các file TSV, train thì cần 3 file, train, dev, test
if args.dataset_path is not None:
    train = "train"
    test = "test"
    dataset_path1 = os.path.join(args.dataset_path, f"{train}.tsv") 
    dataset_path2 = os.path.join(args.dataset_path, f"{test}.tsv")
    logger.info(f"dataset: {args.dataset} at {dataset_path1}")
    config.learning_config.train_dataset_config.data_paths = [dataset_path1]
    config.learning_config.test_dataset_config.data_paths = [dataset_path2]
else:
    raise ValueError("specify the manifest file path using --dataset_path")

test_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.test_dataset_config)
)


#---------------------------
train_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.train_dataset_config)
)
#-------------------------------------------------

# Định nghĩa lại model
conformer = ConformerCtc(
    **config.model_config, 
    vocabulary_size=text_featurizer.num_classes,
)
conformer.encoder.trainable = False
print(text_featurizer.num_classes)

conformer.make(speech_featurizer.shape)

if args.saved:
    conformer.load_weights(args.saved, by_name=True)
else:
    logger.warning("Model is initialized randomly, please use --saved to assign checkpoint")
conformer.summary(line_length=100)

# Đóng băng toàn bộ layers trừ decoder (CTC head)
for layer in conformer.layers:
    if "encoder" in layer.name.lower():
        layer.trainable = False
    else:
        layer.trainable = True  # giữ lại trainable cho output layer
        
conformer.add_featurizers(speech_featurizer, text_featurizer)

batch_size = args.bs or config.learning_config.running_config.batch_size
test_data_loader = test_dataset.create(batch_size)

#-----------------------
train_data_loader = train_dataset.create(batch_size)

blank_id = text_featurizer.blank

true_decoded = []
pred_decoded = []
beam_decoded = []

#for batch in enumerate(test_data_loader):
for k, batch in tqdm(enumerate(test_data_loader)):
    labels, labels_len = batch[1]['labels'], batch[1]['labels_length']

    outputs = conformer(batch[0], training=False)
    logits, logits_len = outputs['logits'], outputs['logits_length']
    probs = softmax(logits)

    if args.beam_size is not None:
        beam = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits, perm=[1, 0, 2]), logits_len, beam_width=args.beam_size, top_paths=1,
        )
        beam = tf.sparse.to_dense(beam[0][0]).numpy()

    for i, (p, l, label, ll) in enumerate(zip(probs, logits_len, labels, labels_len)):
        # p: length x characters
        pred = p[:l].argmax(-1)
        decoded_prediction = []
        previous = blank_id

        # remove the repeting characters and the blanck characters
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

wer_metric = datasets.load_metric("wer")
logger.info(f"Length decoded: {len(true_decoded)}")
logger.info(f"WER: {wer_metric.compute(predictions=pred_decoded, references=true_decoded)}")

if args.beam_size is not None:
    logger.info(f"WER-beam: {wer_metric.compute(predictions=beam_decoded, references=true_decoded)}")


if args.output is not None:
    with file_util.save_file(file_util.preprocess_paths(args.output)) as filepath:
        overwrite = True
        if tf.io.gfile.exists(filepath):
            overwrite = input(f"Overwrite existing result file {filepath} ? (y/n): ").lower() == "y"
        if overwrite:
            logger.info(f"Saving result to {args.output} ...")
            with open(filepath, "w") as openfile:
                openfile.write("PATH\tDURATION\tGROUNDTRUTH\tGREEDY\tBEAMSEARCH\n")
                progbar = tqdm(total=test_dataset.total_steps, unit="batch")
                for i, (groundtruth, greedy) in enumerate(zip(true_decoded, pred_decoded)):
                    openfile.write(f"N/A\tN/A\t{groundtruth}\t{greedy}\tN/A\n")
                    progbar.update(1)
                progbar.close()
def train_model(model, optimizer, train_wavs, train_texts, test_wavs, test_texts, dev_wavs, dev_texts, epochs= 100, batch_size=64):
    if model is None or optimizer is None:
        print("Model or optimizer is None, cannot update learning rate.")
        return
    no_improvement_epochs = 0
    best_loss = float('inf')
    
    train_losses = []
    test_losses = []
    
    with tf.device(device_name):
        epoch_range = []
        test_wers = []
        test_wers_dev = []
        
        for e in range(start_epoch, epochs):
            start_time = time.time()
            
            len_train = len(train_wavs)
            len_test = len(test_wavs)
            len_test_dev = len(dev_wavs)
            train_loss = 0
            test_loss = 0
            train_batch_count = 0
            test_batch_count = 0
            test_batch_count_dev = 0
            wer_results = []
            wer_results_dev = []
            
            current_lr = optimizer.lr.numpy()
            print("Learning Rate at Epoch {}: {:.6f}".format(e + 1, current_lr))
            
            print("Training epoch: {}", format(e+1))
            for start in tqdm(range(0, len_train, batch_size), dynamic_ncols=True):
                
                end = None
                if start + batch_size < len_train
                    end = start + batch_size
                else:
                    end = len_train
                
            