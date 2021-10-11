import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import numpy as np
import tensorflow_addons as tfa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

max_num_visit = 200
num_code = 506


class Config(object):
    def __init__(self):
        self.vocab_dim = num_code
        self.lstm_dim = 256
        self.n_layer = 3
        self.batchsize = 192


def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.2
    else:
        dropout_rate = 0.0
    mask = tf.nn.dropout(tf.ones([inputs.shape[0],1,inputs.shape[2]],dtype=tf.float32), dropout_rate)
    mask = tf.tile(mask, [1,inputs.shape[1],1])
    return inputs*mask
    # b*t*u


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        lstm = DropConnectLSTM
        self.layer = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(config.n_layer)]

    def call(self, x, is_training):
        for layer in self.layer:
            layer.set_mask(is_training)

        for i in range(config.n_layer):
            x = locked_drop(x, is_training)
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self,is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units,self.units*4]),0.2)
        else:
            self.mask = tf.ones([self.units,self.units*4])

    def _process_batch(self, inputs, initial_state):
        if not self.time_major:
            inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
        input_h = initial_state[0]
        input_c = initial_state[1]
        input_h = array_ops.expand_dims(input_h, axis=0)
        input_c = array_ops.expand_dims(input_c, axis=0)

        params = recurrent_v2._canonical_to_params(  # pylint: disable=protected-access
            weights=[
                self.kernel[:, :self.units],
                self.kernel[:, self.units:self.units * 2],
                self.kernel[:, self.units * 2:self.units * 3],
                self.kernel[:, self.units * 3:],
                self.recurrent_kernel[:, :self.units]*self.mask[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2]*self.mask[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3]*self.mask[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:]*self.mask[:, self.units * 3:],
            ],
            biases=[
                self.bias[:self.units],
                self.bias[self.units:self.units * 2],
                self.bias[self.units * 2:self.units * 3],
                self.bias[self.units * 3:self.units * 4],
                self.bias[self.units * 4:self.units * 5],
                self.bias[self.units * 5:self.units * 6],
                self.bias[self.units * 6:self.units * 7],
                self.bias[self.units * 7:],
            ],
            shape=self._vector_shape)

        outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
            inputs,
            input_h=input_h,
            input_c=input_c,
            params=params,
            is_training=True)

        if self.stateful or self.return_state:
            h = h[0]
            c = c[0]
        if self.return_sequences:
            if self.time_major:
                output = outputs
            else:
                output = array_ops.transpose(outputs, perm=(1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h, c]


class Embedding(tf.keras.Model):
    def __init__(self):
        super(Embedding, self).__init__()
        self.linear_forward = tf.keras.layers.Dense(config.lstm_dim)
        self.linear_backward = tf.keras.layers.Dense(config.lstm_dim)
        self.mlp0 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(256)
        self.encoding_feature = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=num_code+1,
                                                                                            output_mode='binary',dtype=tf.float32)
        self.encoding_target = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=283 + 1,
                                                                                            output_mode='binary',dtype=tf.float32)

    def call(self, code, others, length, target):
        code = tf.reshape(code, [-1, code.shape[-1]])
        code = tf.reshape(self.encoding_feature(code), [config.batchsize, max_num_visit, num_code+1])

        mask = tf.sequence_mask(length - 2, max_num_visit - 2)
        code_ = tf.boolean_mask(code[:,1:-1], mask)
        sign = tf.boolean_mask(others[:, 1:-1, 37:], mask)
        code_ = tf.concat((code_, sign), axis=-1)

        left = tf.boolean_mask(others[:, 1:-1, 27:37], mask)
        right = tf.boolean_mask(others[:, 2:, 27:37], mask)
        interval_ = tf.concat((left, right), axis=-1)
        stay = tf.boolean_mask(others[:, 1:-1, 19:27], mask)
        x = tf.concat((code_, interval_, stay), axis=-1)
        feature_vec = self.mlp1(self.mlp0(x))
        feature_vec = feature_vec / tf.math.sqrt(tf.reduce_sum(feature_vec ** 2, axis=-1, keepdims=True))

        target = tf.gather(1 - self.encoding_target(target),weight_idx,axis=1)
        target = tf.tile(tf.expand_dims(target,-2),[1,max_num_visit,1])
        code = tf.concat((code, target),axis=-1)
        interval = others[:,:,27:37]
        interval_backward = tf.concat((tf.zeros((code.shape[0],1,10), dtype=tf.float32),
                                       tf.reverse_sequence(interval, length, seq_axis=1)[:,:-1]),axis=1)

        feature = tf.concat((code, others[:,:,:27], others[:,:,37:]),axis=-1)
        feature_backward = tf.reverse_sequence(feature, length, seq_axis=1)

        x_forward = self.linear_forward(tf.concat((feature, interval),axis=-1)[:,:-2])
        x_backward = self.linear_backward(tf.concat((feature_backward, interval_backward),axis=-1)[:,:-2])
        return x_forward, x_backward, feature_vec


class FeatureNet(tf.keras.Model):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.embeddings = Embedding()
        self.lstm_forward = LSTM()
        self.lstm_backward = LSTM()
        self.dense = tf.keras.layers.Dense(256)
        self.mlp0 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(256)

    def call(self, code, others, length, target, is_training=False):
        length = tf.squeeze(length)
        x_forward, x_backward, vec = self.embeddings(code, others, length, target)
        x_forward = self.lstm_forward(x_forward, is_training)

        x_backward = tf.reverse_sequence(self.lstm_backward(x_backward, is_training),length - 2, seq_axis=1)

        x_forward = tf.boolean_mask(x_forward, tf.sequence_mask(length - 2, max_num_visit-2))
        x_backward = tf.boolean_mask(x_backward, tf.sequence_mask(length - 2, max_num_visit - 2))
        x = tf.concat((x_forward, x_backward), axis=-1)
        feature_vec = self.mlp1(self.mlp0(x))
        feature_vec = feature_vec / tf.math.sqrt(tf.reduce_sum(feature_vec ** 2, axis=-1, keepdims=True))
        return feature_vec, vec


def train(p):
    lengths = np.load('../XINMENG/length.npy').astype('int32')
    features = np.load('../XINMENG/code.npy').astype('int32') + 1
    summaries = np.load('../XINMENG/others.npy').astype('float32')
    targets = np.load('../XINMENG/target.npy').astype('int32') + 1

    # train_idx = np.load('../XINMENG/train_idx.npy')
    # train_idx = np.random.choice(train_idx, int(p/10*len(train_idx)))
    # np.save('train_idx_'+str(p),train_idx)
    # val_idx = np.load('../XINMENG/val_idx.npy')
    # val_idx = np.random.choice(val_idx, int(p / 10 * len(val_idx)))
    # np.save('val_idx_' + str(p), val_idx)
    train_idx, val_idx = np.load('train_idx_'+str(p)+'.npy'), np.load('val_idx_'+str(p)+'.npy')

    dataset_train = tf.data.Dataset.from_tensor_slices((features[train_idx], summaries[train_idx],
                                                        lengths[train_idx],targets[train_idx])).shuffle(4096 * 4,
                                                                                     reshuffle_each_iteration=True)
    parsed_dataset_train = dataset_train.batch(config.batchsize, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    dataset_val = tf.data.Dataset.from_tensor_slices((features[val_idx], summaries[val_idx],
                                                      lengths[val_idx], targets[val_idx]))
    parsed_dataset_val = dataset_val.batch(config.batchsize, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    feature_net = FeatureNet()
    checkpoint_directory = "training_checkpoints_bilstm_" + str(p)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=feature_net)
    # checkpoint.restore(checkpoint_prefix + '-20').expect_partial()

    @tf.function
    def one_step(batch, is_training):
        with tf.GradientTape() as tape:
            feature_vec, feature_vec_syn = feature_net(*batch, is_training)
            pair_wise_d = tf.matmul(feature_vec, feature_vec_syn, transpose_b=True) * 10
            loss = tf.linalg.diag_part(tf.nn.log_softmax(pair_wise_d))
            loss = -tf.reduce_mean(loss * (1 - tf.math.exp(loss)) ** 2)
        if is_training:
            grads = tape.gradient(loss, feature_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, feature_net.trainable_variables))
        return loss

    print('training start')
    for epoch in range(400):
        step_val = 0
        step_train = 0

        start_time = time.time()
        loss_val = 0
        loss_train = 0

        for batch_sample in parsed_dataset_train:
            step_loss = one_step(batch_sample, True).numpy()
            loss_train += step_loss
            step_train += 1

        for batch_sample in parsed_dataset_val:
            step_loss = one_step(batch_sample, False).numpy()
            loss_val += step_loss
            step_val += 1

        duration_epoch = int(time.time() - start_time)
        format_str = 'epoch: %d, train_loss = %f, val_loss = %f (%d)'
        print(format_str % (epoch, loss_train / step_train, loss_val / step_val,
                            duration_epoch))
        if epoch % 20 == 19:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    config = Config()
    weight_idx = tf.constant(np.arange(283)[np.load('../AUG2/label_weight.npy') >= 50], dtype=tf.int32)
    train(10)