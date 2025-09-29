# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import abc
from typing import Union
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..featurizers.speech_featurizers import (
    load_and_convert_to_wav, 
    convert_waveform_to_encoded_wav,
    read_raw_audio, 
    tf_read_raw_audio, 
    TFSpeechFeaturizer
)
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils import feature_util, file_util, math_util, data_util

logger = tf.get_logger()


BUFFER_SIZE = 10000
AUTOTUNE = tf.data.experimental.AUTOTUNE


class BaseDataset(metaclass=abc.ABCMeta):
    """ Based dataset for all models """

    def __init__(
        self,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        indefinite: bool = False,
        drop_remainder: bool = True,
        stage: str = "train",
        **kwargs,
    ):
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError('data_paths must be a list of string paths')
        self.cache = cache  # whether to cache transformed dataset to memory
        self.shuffle = shuffle  # whether to shuffle tf.data.Dataset
        if buffer_size <= 0 and shuffle:
            raise ValueError("buffer_size must be positive when shuffle is on")
        self.buffer_size = buffer_size  # shuffle buffer size
        self.stage = stage  # for defining tfrecords files
        self.drop_remainder = drop_remainder  # whether to drop remainder for multi gpu training
        self.indefinite = indefinite  # Whether to make dataset repeat indefinitely -> avoid the potential last partial batch
        self.total_steps = None  # for better training visualization

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create(self, batch_size):
        raise NotImplementedError()


class ASRDataset(BaseDataset):
    """ Dataset for ASR using Generator """

    def __init__(
        self,
        stage: str,
        speech_featurizer: TFSpeechFeaturizer,
        text_featurizer: TextFeaturizer,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        indefinite: bool = False,
        drop_remainder: bool = True,
        buffer_size: int = BUFFER_SIZE,
        input_padding_length: int = 800, #Số lượng frame fixed
        label_padding_length: int = 200,   #Độ dài label.
        **kwargs,
    ):
        super().__init__(
            data_paths=data_paths, 
            cache=cache, shuffle=shuffle, stage=stage, buffer_size=buffer_size,
            drop_remainder=drop_remainder, indefinite=indefinite
        )
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.input_padding_length = input_padding_length
        self.label_padding_length = label_padding_length


    # -------------------------------- ENTRIES -------------------------------------

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0: return
        self.entries = []
        # Duyệt qua tất cả các tệp trong self.data_paths
        for file_path in self.data_paths:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                # Đọc tất cả các dòng trong tệp và tách thành các phần tử trong danh sách.
                temp_lines = f.read().splitlines()
                # Skip the header of tsv file
                self.entries += temp_lines[1:]
        # The files is "\t" seperated
        #Tách dòng thành ba phần với tối đa 2 lần tách
        self.entries = [line.split("\t", 2) for line in self.entries]
        #Chuyển transcript thành chuỗi các chỉ số, đặc trưng văn bản.
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join([str(x) for x in self.text_featurizer.extract(line[-1]).numpy()])
        #Chuyển đổi tất cả các dòng thành một dạng đã tiền xử lý.
        self.entries = np.array(self.entries)
        if self.shuffle: np.random.shuffle(self.entries)  # Mix transcripts.tsv
        self.total_steps = len(self.entries)  #Cập nhật bước huấn luyện 
# Output [
#   ['data/audio2.wav', '2.80', '8 15 23 0 1 18 5 0 25 15 21'],
#   ['data/audio3.wav', '1.95', '14 9 3 5 0 20 15 0 13 5 5 20 0 25 15 21']
# ]

    # -------------------------------- LOAD AND PREPROCESS -------------------------------------

    def generator(self):
        for path, _, indices in self.entries:
            audio = load_and_convert_to_wav(path).numpy()
            yield bytes(path, "utf-8"), audio, bytes(indices, "utf-8")


    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)
            features = self.speech_featurizer.tf_extract(signal)
            #features.shape = [T, F, 1]
            input_length = tf.cast(tf.shape(features)[0], tf.int32)   #T

            label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32) #indices
            label_length = tf.cast(tf.shape(label)[0], tf.int32)                       # len of indices

            prediction = self.text_featurizer.prepand_blank(label)                     
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)

            return path, features, input_length, label, label_length, prediction, prediction_length

    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        data = self.tf_preprocess(path, audio, indices)
        _, features, input_length, label, label_length, prediction, prediction_length = data # remove path.
        return (
            data_util.create_inputs(
                inputs=features,
                inputs_length=input_length,
                predictions=prediction,
                predictions_length=prediction_length
            ),
            data_util.create_labels(
                labels=label,
                labels_length=label_length
            )
        )
    # Xử lý dataset để chuẩn bị dữ liệu đầu vào cho quá trình train/ suy luận.
    # Biến đổi dataset thành 1 dataset có batch, đã padding, shuffle, cache, repeat và prefect.
    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        self.total_steps = math_util.get_num_batches(self.total_steps, batch_size, drop_remainders=self.drop_remainder) # gán bằng 1 luôn
        if self.cache:
            dataset = dataset.cache()

        #Trộn ngẫu nhiên dataset, trộn lại dữ liệu sau mỗi epoch.
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        #Số bước lặp lại dựa vào số batch, ko cần cho infer
        if self.indefinite and self.total_steps:
            dataset = dataset.repeat()
        # Gom các N mẫu lại thành batch, tự động padding các tensor để có cùng kích thước với phần tử dài nhất trong batch.
        max_input_len = 0
        count_above_400 = 0
        max_label_len = 0

        for element in dataset:
            input_tensor = element[0]['inputs']
            label_tensor = element[1]['labels']
            input_len = input_tensor.shape[0]
            label_len = label_tensor.shape[0]
            
            # Cập nhật max_input_len
            if input_len > max_input_len:
                max_input_len = input_len
            # Cập nhật max độ dài label
            if label_len > max_label_len:
                max_label_len = label_len
            
            # Kiểm tra input_len có lớn hơn 1600 không
            if input_len > 800:
                count_above_400 += 1

        print(f"🔍 Max input length (T): {max_input_len}")
        print(f"🔍 Number of inputs with length > 700: {count_above_400}")
        print(f"🔍 Max label length: {max_label_len}")
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            #Chỉ định hình dạng từng phần sau khi padding
            padded_shapes=(
                data_util.create_inputs(
                    inputs=tf.TensorShape([self.input_padding_length, 80, 1]),
                    inputs_length=tf.TensorShape([]),  #Độ dài thật của input, a single value
                    #Đầu ra mong muốn model học theo.
                    predictions=tf.TensorShape([self.label_padding_length]),
                    predictions_length=tf.TensorShape([])  #Độ dài thật của chuỗi dự đoán.
                    #Cần độ dài thật vì khi tính CTC, chúng ko tính phần padding, padding để đưa làm input model
                ),
                #ko cần cái này.
                data_util.create_labels(
                    labels=tf.TensorShape([self.label_padding_length]),
                    labels_length=tf.TensorShape([])
                ),
            ),
            # Gía trị dùng để padding.
            padding_values=(
                data_util.create_inputs(
                    inputs=0.0,  #Cho spectrogram đầu vào. Tất cả frame được padding = 0.
                    inputs_length=0, #ko cần padding chiều dài thực tế.
                    predictions=self.text_featurizer.blank, #padding ký tự trống.
                    predictions_length=0   #Chiều dài label thực tế, ko cần.
                ),
                data_util.create_labels(
                    labels=self.text_featurizer.blank, #same
                    labels_length=0                 #Same.
                )
            ),
            # có bỏ phần tử cuối cùng hay không.
            drop_remainder=self.drop_remainder
        )

        # PREFETCH to improve speed of input length
        # prefectch: tải trước batch tiếp theo khi batch hiện tại đang xử lý.
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return print("Couldn't create")
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
        )
        return self.process(dataset, batch_size)


class ASRSliceDataset(ASRDataset):
    """ Dataset for ASR using Slice """

    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes): return load_and_convert_to_wav(path.decode("utf-8")).numpy()
        #record[0] là path đến file âm thanh
        audio = tf.numpy_function(fn, inp=[record[0]], Tout=tf.string)
        #audio: tensor: dữ liệu âm thanh đã được mã hóa lại dạng byte string của wav
        return record[0], audio, record[2]
    # path, audio, text

    #Chia dữ liệu thành các batch với kích thước tương ứng.
    def create(self, batch_size: int):
        self.read_entries() # độc tệp TSV, tách dữ liệu thành 3 phần, tiền xử lý transcript và lưu kq vào self.entrices
        if not self.total_steps or self.total_steps == 0: return None
        # biến đổi entrices thành các phần tử riêng biệt
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)

        #áp dụng hàm self.load cho từng phần tử trong dataset.
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        return self.process(dataset, batch_size)
    #Outputs : features, input_lengths, labels, label_lengths, pred_inp
    def preprocess_from_mic(self, waveform: np.ndarray):
        """
        Tiền xử lý một đoạn âm thanh lấy từ micro (Tensor đã chuẩn hóa, float32, 16Khz)
        Trả về dict phù hợp để đưa vào model."""
        # Chuyển text về indices.
        def fn(waveform: np.ndarray): return convert_waveform_to_encoded_wav(waveform, orig_sr=16000).numpy()
        audio_tensor = tf.numpy_function(fn, inp=[waveform], Tout=tf.string)
        signal = tf_read_raw_audio(audio_tensor, self.speech_featurizer.sample_rate)
        features = self.speech_featurizer.tf_extract(signal)
        #features.shape = [T, F, 1]
        input_length = tf.cast(tf.shape(features)[0], tf.int32)   #T
        # dataset = dataset.padded_batch(
        #     batch_size=1,  # Dù bạn không muốn batch size, nhưng `batch_size` là tham số bắt buộc
        #     padded_shapes=(
        #         data_util.create_inputs(
        #             inputs=tf.TensorShape([self.input_padding_length, 80, 1]),  # padding cho spectrogram
        #             inputs_length=tf.TensorShape([]),  # độ dài thực tế của input
        #         ),
        #     ),
        #     padding_values=(
        #         data_util.create_inputs(
        #             inputs=0.0,  # padding cho spectrogram, 0.0 là giá trị padding cho spectrogram
        #             inputs_length=0,  # padding cho inputs_length, có thể là 0 hoặc một giá trị phù hợp
        #         ),
        #     ),
        #     drop_remainder=self.drop_remainder  # Nếu bạn muốn loại bỏ batch không đầy
        # )
        # Thực hiện padding thủ công cho spectrogram
        padded_input = tf.pad(features, [[0, self.input_padding_length - tf.shape(features)[0]], [0, 0], [0, 0]], constant_values=0.0)

        dataset = data_util.create_inputs(
            inputs=padded_input,
            inputs_length=input_length,
        )
        # Tạo tuple (inputs, inputs_length)
        inputs_tensor = tf.convert_to_tensor(padded_input, dtype=tf.float32)
        inputs_length_tensor = tf.convert_to_tensor(input_length, dtype=tf.int32)

        return inputs_tensor, inputs_length_tensor

        #return dataset

    def preprocess_dataset(self, tfrecord_path, shard_size=0, max_len=None):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        logger.info(f"Preprocess dataset")
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        self.create_preprocessed_tfrecord(dataset, tfrecord_path, shard_size, max_len)
