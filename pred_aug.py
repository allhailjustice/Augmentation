import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os,pickle
import numpy as np
import argparse
import tensorflow_addons as tfa


class Config(object):
    def __init__(self):
        self.embedding_dim = 128
        self.lstm_dim = 384
        self.n_layer = 3
        self.batchsize = batchsize
        self.num_dx = 181


def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.25
    else:
        dropout_rate = 0.0
    mask = tf.nn.dropout(tf.ones([inputs.shape[0], 1, inputs.shape[2]], dtype=tf.float32), dropout_rate)
    mask = tf.tile(mask, [1, inputs.shape[1], 1])
    return inputs * mask
    # b*t*u


class SingleLSTM(tf.keras.Model):
    def __init__(self):
        super(SingleLSTM, self).__init__()
        lstm = DropConnectLSTM
        self.layer = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(config.n_layer)]

    def call(self, x, is_training):
        for layer in self.layer:
            layer.set_mask(is_training)

        for k in range(config.n_layer):
            x = locked_drop(x, is_training)
            x = self.layer[k](x)
            x = self.layer_norm[k](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self, is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units, self.units * 4]), 0.25)
        else:
            self.mask = tf.ones([self.units, self.units * 4])

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
                self.recurrent_kernel[:, :self.units] * self.mask[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2] * self.mask[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3] * self.mask[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:] * self.mask[:, self.units * 3:],
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
        self.linear = tf.keras.layers.Dense(config.embedding_dim)
        self.encoding = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=num_code+1,
                                                                                    output_mode='binary',dtype=tf.float32)

    def call(self, code, others):
        code = tf.reshape(code, [-1, code.shape[-1]])
        code = tf.reshape(self.encoding(code), [config.batchsize, max_num_visit, num_code+1])
        output = self.linear(tf.concat((code, others), axis=-1))
        return output


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.embeddings = Embedding()
        self.lstm = SingleLSTM()
        self.proj1 = tf.keras.layers.Dense(config.lstm_dim)
        self.proj2 = tf.keras.layers.Dense(181, activation=tf.nn.sigmoid)

    def call(self, code, others, length, is_training=False):
        x = self.embeddings(code, others)
        x = self.proj1(x)
        x = self.lstm(x, is_training)
        x = tf.gather_nd(x, tf.concat(
            (tf.expand_dims(tf.range(batchsize, dtype=tf.int32), -1), tf.expand_dims(length - 1, -1)), axis=-1))
        output = tf.squeeze(self.proj2(x))

        return output


def train():
    optimizer = tfa.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.0)
    model = Model()
    m = [tf.keras.metrics.AUC(num_thresholds=200) for _ in range(181)]
    encoding = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=283+1,
                                                                           output_mode='binary', dtype=tf.float32)

    @tf.function
    def one_step(feature, summary, target, target_range, length, is_training):
        length = tf.squeeze(length)
        target = tf.gather(encoding(target)[:,1:], weight_idx, axis=1)
        target_range = tf.gather(1 - encoding(target_range)[:,1:], weight_idx, axis=1)
        with tf.GradientTape() as tape:
            output = model(feature, summary, length, is_training)
            loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(target, output) * target_range) / tf.reduce_sum(target_range)
        if is_training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        for k in range(181):
            m[k].update_state(tf.boolean_mask(target[:, k], target_range[:, k]),
                              tf.boolean_mask(output[:, k], target_range[:, k]))
        return loss

    min_val_loss = 10000
    max_auc = 0
    print('training start')
    patience = 0
    for epoch in range(150):
        history_train = np.load('syn/history_train_'+str(epoch%25)+'.npy').astype('int32')
        with open('syn/code_train_' + str(epoch%25) + '.pkl', 'rb') as file:
            feature_train = pickle.load(file)
        feature_train = np.array([np.concatenate((np.array(f), np.zeros((200 - len(f), 101))), axis=0) for f in feature_train],dtype='int32')
        with open('syn/others_train_' + str(epoch%25) + '.pkl', 'rb') as file:
            summary_train = pickle.load(file)
        summary_train = np.array(
            [np.concatenate((np.array(s), -np.ones((200 - len(s), 39))), axis=0) for s in summary_train],dtype='float32')
        dataset_train = tf.data.Dataset.from_tensor_slices(
            (feature_train, summary_train, target_train,
             history_train,
             length_train)).shuffle(4096 * 8, reshuffle_each_iteration=True)
        parsed_dataset_train = dataset_train.batch(batchsize, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)

        t = time.time()
        loss_train = 0
        step_train = 0
        for mi in m:
            mi.reset_states()
        for batch in parsed_dataset_train:
            loss_train += one_step(*batch, True).numpy()
            step_train += 1
        auc_train = np.mean(np.array([mi.result().numpy() for mi in m]))

        loss_val = 0
        step_val = 0
        for batch in parsed_dataset_val:
            loss_val += one_step(*batch, False).numpy()
            step_val += 1

        for mi in m:
            mi.reset_states()
        for batch in parsed_dataset_test:
            _ = one_step(*batch, False)
        auc_test = np.mean(np.array([mi.result().numpy() for mi in m]))

        format_str = 'epoch: %d, train_loss = %f, val_loss = %f, train_auc = %f, test_auc = %f (%d)'
        print(format_str % (epoch, loss_train / step_train,
                            loss_val / step_val, auc_train, auc_test, (time.time() - t)))

        if loss_val / step_val < min_val_loss:
            patience = 0
            min_val_loss = loss_val / step_val
            max_auc = np.array([mi.result().numpy() for mi in m])
        else:
            patience += 1
        if patience >= 15 and epoch > 60:
            break
    return max_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=str)
    parser.add_argument('exp', type=int)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    NUM_GPU = 1
    batchsize = 192
    num_code = 506
    max_num_visit = 200
    p = args.exp
    config = Config()
    weight_idx = tf.constant(np.arange(283)[np.load('../AUG2/label_weight.npy') >= 50], dtype=tf.int32)
    train_idx, test_idx, val_idx = np.load('train_idx_' + str(p) + '.npy'), np.load(
        '../XINMENG/test_idx.npy'), np.load('val_idx_' + str(p) + '.npy')
    lengths, targets,  = np.load('../XINMENG/length.npy').astype('int32'), \
                         np.load('../XINMENG/target.npy').astype('int32') + 1
    features = np.load('../XINMENG/code.npy').astype('int32') + 1
    summaries = np.load('../XINMENG/others.npy').astype('float32')
    histories = np.load('../XINMENG/history.npy').astype('int32') + 1
    dataset_val = tf.data.Dataset.from_tensor_slices((features[val_idx], summaries[val_idx], targets[val_idx],
                                                      histories[val_idx],
                                                      lengths[val_idx]))
    parsed_dataset_val = dataset_val.batch(batchsize, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    dataset_test = tf.data.Dataset.from_tensor_slices((features[test_idx], summaries[test_idx], targets[test_idx],
                                                       histories[test_idx],
                                                       lengths[test_idx]))
    parsed_dataset_test = dataset_test.batch(batchsize, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)
    length_train = lengths[train_idx]
    target_train = targets[train_idx]
    order = np.argsort(length_train)
    length_train = length_train[order]
    target_train = target_train[order]

    auc = []
    for i in range(20):
        auc.append(train())
    np.save('result/aug_'+str(p), auc)
