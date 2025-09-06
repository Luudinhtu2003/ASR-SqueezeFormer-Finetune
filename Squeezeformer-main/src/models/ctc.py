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

from typing import Dict, Union
import numpy as np
import tensorflow as tf
 
from .base_model import BaseModel
from ..augmentations.augmentation import SpecAugmentation
from ..featurizers.speech_featurizers import TFSpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils import math_util, shape_util, data_util
from ..losses.ctc_loss import CtcLoss

logger = tf.get_logger()

# Khi dùng lại 1 lớp, nó có thể định nghĩa thêm các đối số, 
# dùng lại các hàm hoặc ghi đè lên các hàm đã được định nghĩa.
augmentation_config = {
    "time_masking": {
        "num_masks": 2,
        "p_upperbound": 0.05
    },
    "freq_masking": {
        "num_masks": 1,
        "mask_factor": 10
    }
}
name: str = "conformer"
augmentation = SpecAugmentation(
                num_freq_masks=augmentation_config['freq_masking']['num_masks'],
                freq_mask_len=augmentation_config['freq_masking']['mask_factor'],
                num_time_masks=augmentation_config['time_masking']['num_masks'],
                time_mask_prop=augmentation_config['time_masking']['p_upperbound'],
                name=f"{name}_specaug"
            )
class CtcModel(BaseModel):
    def __init__(
        self,
        # Mô hình mã hóa biến spectrogram thành đặc trưng thời gian
        encoder: tf.keras.Model,
        #Giari mã
        decoder: Union[tf.keras.Model, tf.keras.layers.Layer] = None,
        #Tiền xử lý tăng cường dữ liệu
        augmentation: tf.keras.Model = None,
        vocabulary_size: int = None,
        #Các tham số khác truyền cho lớp cha Base Model 
        **kwargs,
    ):
        super().__init__(**kwargs) #Gọi constructor của lớp cha và truyền các tham số bổ sung.
        self.encoder = encoder
        if decoder is None:
            assert vocabulary_size is not None, "vocabulary_size must be set"
            self.decoder = tf.keras.layers.Dense(units=vocabulary_size, name=f"{self.name}_logits")
        else:
            self.decoder = decoder
        self.augmentation = augmentation
        self.time_reduction_factor = 1 #Hệ so rút gọn thời gian.
	
    #Xây dụng mô hình với input_shape, build sẵn kiến trúc model với đầu vào nào đó. 
    #input_shape: hình dạng của input của model (None, 80, 1)
    # batch_size
    def make(self, input_shape, batch_size=None):
        inputs = tf.keras.Input(input_shape, batch_size=batch_size, dtype=tf.float32)
        #Lưu trữ độ dài thực tế của các chuỗi trong dữ liệu
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self(
            #Tạo đối tượng dữ liệu của mô hình
            data_util.create_inputs(
                inputs=inputs,
                inputs_length=inputs_length
            ),
            training=False
        )

    #Biên dịch mô hình với loss và optimizer.
    def compile(self, optimizer, blank=0, run_eagerly=None, **kwargs):
        loss = CtcLoss(blank=blank)
        super().compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)

    def add_featurizers(
        self,
        speech_featurizer: TFSpeechFeaturizer,
        text_featurizer: TextFeaturizer,
    ): 
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    # Step 3, Xác định forward pass của model
    def call(self, inputs, training=False, **kwargs):
        x, x_length = inputs["inputs"], inputs["inputs_length"]
        # if training and augmentation is not None: #no
        #     x = augmentation(x, x_length)
        
        #[B, T, V1, V2]
        logits = self.encoder(x, x_length, training=training, **kwargs)
        # [B, T, D]
        logits = self.decoder(logits, training=training, **kwargs)
        return data_util.create_logits(
            logits=logits,
            logits_length=math_util.get_reduced_length(x_length, self.time_reduction_factor)
        )
    #Logits shape: (1, 925, 90) (batch_size, time_steps, num_classes)

    # -------------------------------- GREEDY -------------------------------------
    @tf.function
    def recognize_from_logits(self, logits: tf.Tensor, lengths: tf.Tensor):
        probs = tf.nn.softmax(logits)
        # blank is in the first index of `probs`, where `ctc_greedy_decoder` supposes it to be in the last index.
        # threfore, we move the first column to the last column to be compatible with `ctc_greedy_decoder`
        probs = tf.concat([probs[:, :, 1:], tf.expand_dims(probs[:, :, 0], -1)], axis=-1)
        def _map(elems): return tf.numpy_function(self._perform_greedy, inp=[elems[0], elems[1]], Tout=tf.string)

        return tf.map_fn(_map, (probs, lengths), fn_output_signature=tf.TensorSpec([], dtype=tf.string))
    

    @tf.function
    def recognize(self, inputs: Dict[str, tf.Tensor]):
        logits = self(inputs, training=False)
        probs = tf.nn.softmax(logits["logits"])
        # send the first index (skip token) to the last index
        # for compatibility with the ctc_decoders library
        probs = tf.concat([probs[:, :, 1:], tf.expand_dims(probs[:, :, 0], -1)], axis=-1)
        lengths = logits["logits_length"]

        def map_fn(elem): return tf.numpy_function(self._perform_greedy, inp=[elem[0], elem[1]], Tout=tf.string)

        return tf.map_fn(map_fn, [probs, lengths], fn_output_signature=tf.TensorSpec([], dtype=tf.string))

    def _perform_greedy(self, probs: np.ndarray, length):
        from ctc_decoders import ctc_greedy_decoder
        decoded = ctc_greedy_decoder(probs[:length], vocabulary=self.text_featurizer.non_blank_tokens)
        return tf.convert_to_tensor(decoded, dtype=tf.string)

    # -------------------------------- BEAM SEARCH -------------------------------------

    @tf.function
    def recognize_beam(self, inputs: Dict[str, tf.Tensor], lm: bool = False):
        logits = self(inputs, training=False)
        probs = tf.nn.softmax(logits["logits"])

        def map_fn(prob): return tf.numpy_function(self._perform_beam_search, inp=[prob, lm], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def _perform_beam_search(self, probs: np.ndarray, lm: bool = False):
        from ctc_decoders import ctc_beam_search_decoder
        decoded = ctc_beam_search_decoder(
            probs_seq=probs,
            vocabulary=self.text_featurizer.non_blank_tokens,
            beam_size=self.text_featurizer.decoder_config.beam_width,
            ext_scoring_func=self.text_featurizer.scorer if lm else None
        )
        decoded = decoded[0][-1]

        return tf.convert_to_tensor(decoded, dtype=tf.string)
 