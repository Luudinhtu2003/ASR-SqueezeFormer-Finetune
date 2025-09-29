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

import tensorflow as tf
from tensorflow.keras import mixed_precision as mxp
from tensorflow.keras import mixed_precision  
from ..utils import file_util, env_util
 
class BaseModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = {}
        self.use_loss_scale = False

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        with file_util.save_file(filepath) as path:
            super().save(
                filepath=path,
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces,
            )

    def save_weights(
        self,
        filepath,
        overwrite=True,
        save_format=None,
        options=None,
    ):
        with file_util.save_file(filepath) as path:
            super().save_weights(
                filepath=path,
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )

    def load_weights(
        self,
        filepath,
        by_name=False,
        skip_mismatch=False,
        options=None,
    ):
        with file_util.read_file(filepath) as path:
            super().load_weights(
                filepath=path,
                by_name=by_name,
                skip_mismatch=skip_mismatch,
                options=options,
            )

    @property
    def metrics(self):
        return self._metrics.values()

    # Thêm các metric mong muốn
    def add_metric(self, metric: tf.keras.metrics.Metric):
        self._metrics[metric.name] = metric

    def make(self, *args, **kwargs):
        """ Custom function for building model (uses self.build so cannot overwrite that function) """
        raise NotImplementedError()

    #Khai báo đầy đủ cấu hình huấn luyện: loss, optimizer, metrics.
    """
    - Nếu không dùng TPU, mô hình sẽ:
    - Dùng `mixed_float16` để tăng tốc training trên GPU.
    - Dùng `LossScaleOptimizer` để tránh lỗi số học với float16.
    - Ghi lại metric loss trung bình theo `tf.keras.metrics.Mean`.
    - Cuối cùng gọi `super().compile()` để hoàn tất việc cấu hình mô hình.
    """
    def compile(self, loss, optimizer, run_eagerly=None, **kwargs):
        if not env_util.has_devices("TPU"):
            # Đặt policy mixed precision cho toàn bộ mô hình
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)

            # Tạo optimizer với mixed precision mà không cần loss_scale
            optimizer = mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.get(optimizer)
            )
            self.use_loss_scale = True
        #Tạo 1 mean để theo dõi giá trị trung bình loss sau mỗi batch hoặc epoch, lưu vào self._metrics
        loss_metric = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        self._metrics = {loss_metric.name: loss_metric}
        #Gọi compile để hoàn tất cấu hình
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)


    # -------------------------------- STEP FUNCTIONS -------------------------------------

    # Nhận 2 đầu vào là input (spectrogram) và y_true (nhãn thật)
    #Step 2.
    @tf.function
    def gradient_step(self, inputs, y_true):

        #(['inputs', 'inputs_length', 'predictions', 'predictions_length'])
        with tf.GradientTape() as tape:
            #(8, 3700, 80, 1)
            y_pred = self(inputs, training=True) #Gọi call 
            # [B, T2, dmodel]
            loss = self.loss(y_true, y_pred)
            if self.use_loss_scale:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        #Tính gradient. Nếu có mixed precision thì tính của scaled_loss rồi unscale lại
        if self.use_loss_scale:
            gradients = tape.gradient(scaled_loss, self.trainable_weights)
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        else:
            gradients = tape.gradient(loss, self.trainable_weights)
        # num_params_with_grad = sum([tf.reduce_prod(var.shape).numpy() for var, grad in zip(self.trainable_weights, gradients) if grad is not None])
        # print(f"Số lượng tham số được cập nhật gradient: {num_params_with_grad}")

        return loss, y_pred, gradients
    
    @tf.function
    def train_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of training data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric
 
        """
        inputs, y_true = batch  #batch: 1 batch từ tf.data.Dataset/
        #Thực hiện lan truyền ngược
        loss, y_pred, gradients = self.gradient_step(inputs, y_true)
        #Cập nhật trọng số.
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._metrics["loss"].update_state(loss)
        if 'step_loss' in self._metrics:
            self._metrics['step_loss'].update_state(loss)
        if 'WER' in self._metrics:
            self._metrics['WER'].update_state(y_true, y_pred)
        if 'labels' in self._metrics:
            self._metrics['labels'].update_state(y_true)
        if 'logits' in self._metrics:
            self._metrics['logits'].update_state(y_pred)
        if 'logits_len' in self._metrics:
            self._metrics['logits_len'].update_state(y_pred)
        
        return {m.name: m.result() for m in self.metrics}
#Trả về kết quả của các metric dưới dạng {metric_name: giá trị} 
# – dùng để log hoặc hiển thị tiến trình huấn luyện.
# 
#     Gợi ý dùng:
#     for epoch in range(num_epochs):
#     for batch in train_dataset:
#         logs = model.train_step(batch)
#         print(logs)
#    
    def test_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]: a batch of validation data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric prefixed with "val_"

        """
        inputs, y_true = batch
        y_pred = self(inputs, training=False)
        loss = self.loss(y_true, y_pred)
        self._metrics["loss"].update_state(loss)
        if 'step_loss' in self._metrics:
            self._metrics['step_loss'].update_state(loss)
        if 'WER' in self._metrics:
            self._metrics['WER'].update_state(y_true, y_pred)
        if 'labels' in self._metrics:
            self._metrics['labels'].update_state(y_true)
        if 'logits' in self._metrics:
            self._metrics['logits'].update_state(y_pred)
        if 'logits_len' in self._metrics:
            self._metrics['logits_len'].update_state(y_pred)
        # Lấy kết quả từ các metric và trả thêm y_pred, y_true nếu cần
        result = {m.name: m.result() for m in self.metrics}
        result["y_pred"] = y_pred
        result["y_true"] = y_true

        return result

    def predict_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of testing data

        Returns:
            [tf.Tensor]: stacked tensor of shape [B, 3] with each row is the text [truth, greedy, beam_s_search]
        """
        inputs, y_true = batch
        # Chuyển nhãn số thành văn bản (dạng string tensor)
        labels = self.text_featurizer.iextract(y_true["labels"])
        greedy_decoding = self.recognize(inputs)
        if self.text_featurizer.decoder_config.beam_width == 0:
            beam_search_decoding = tf.map_fn(lambda _: tf.convert_to_tensor("", dtype=tf.string), labels)
        else:
            beam_search_decoding = self.recognize_beam(inputs)
        return tf.stack([labels, greedy_decoding, beam_search_decoding], axis=-1)
        #Gộp tất cả thành dạng tensor [B, 3]. Cột 0: ground truth, greedy decode, beam search decode.
    # -------------------------------- INFERENCE FUNCTIONS -------------------------------------

    def recognize(self, *args, **kwargs):
        """ Greedy decoding function that used in self.predict_step """
        raise NotImplementedError()

    def recognize_beam(self, *args, **kwargs):
        """ Beam search decoding function that used in self.predict_step """
        raise NotImplementedError()

