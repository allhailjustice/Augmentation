import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import numpy as np
import argparse


class Config(object):
    def __init__(self):
        self.embedding_dim = 128
        self.lstm_dim = 384
        self.n_layer = 3
        self.batchsize = batchsize
        # number of diagnosis codes to be predicted (the target)
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


# no need to read this part, this is only a revised version of LSTM
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
    optimizer = tf.optimizers.Adam(learning_rate=5e-5)
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
                            loss_val / step_val, auc_train, auc_test, (time.time() - t) / 60))

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

    # total number of codes in the VUMC data, 262 CCS diagnosis codes and 244 CCS procedure code
    # note, there are 282 diagnosis (0-281) in total in the dictionary
    num_code = 506

    # we limited the number of visits of a patient record to 200
    max_num_visit = 200

    config = Config()

    # of all diagnosis codes, we only select those with at least 50 occurrences in the dataset as our target
    weight_idx = tf.constant(np.arange(283)[np.load('../AUG2/label_weight.npy') >= 50], dtype=tf.int32)
    lengths, features, targets, histories, summaries = np.load('length.npy').astype('int32'), \
                                                       np.load('code.npy').astype('int32') + 1, \
                                                       np.load('target.npy').astype('int32') + 1, \
                                                       np.load('history.npy').astype('int32') + 1, \
                                                       np.load('others.npy').astype('float32')
    # "features" is code data
    # "lengths" is the number of visits for each patient
    # "summaries" is demographics data and visit details
    # "targets" is the label data we try to predict
    # "histories" is the codes patient used to be diagnosed prior to the targets, we don't include them as the candidate for prediction
    # code=-1 implies empty, we add 1 to each code to make them all positive

    train_idx, test_idx, val_idx = np.load('train_idx_'+str(p)+'.npy'), np.load('test_idx.npy'), np.load('val_idx_'+str(p)+'.npy')

    dataset_train = tf.data.Dataset.from_tensor_slices((features[train_idx], summaries[train_idx], targets[train_idx],
                                                        histories[train_idx],
                                                        lengths[train_idx])).shuffle(4096*8,
                                                                                     reshuffle_each_iteration=True)
    parsed_dataset_train = dataset_train.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    dataset_val = tf.data.Dataset.from_tensor_slices((features[val_idx], summaries[val_idx], targets[val_idx],
                                                      histories[val_idx],
                                                      lengths[val_idx]))
    parsed_dataset_val = dataset_val.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    dataset_test = tf.data.Dataset.from_tensor_slices((features[test_idx], summaries[test_idx], targets[test_idx],
                                                       histories[test_idx],
                                                       lengths[test_idx]))
    parsed_dataset_test = dataset_test.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)
    del features, summaries, histories, targets, lengths

    auc = []
    for i in range(20):
        auc.append(train())
    np.save('result/total', auc)
