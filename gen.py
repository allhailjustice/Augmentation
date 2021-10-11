import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import numpy as np
import pickle
import operator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_GPU = 1
num_code = 506
max_num_visit = 200


class Config(object):
    def __init__(self):
        self.vocab_dim = num_code
        self.lstm_dim = 256
        self.n_layer = 3
        self.batchsize = 20
        self.Z_DIM = 128
        self.G_DIMS = [256, 256, 256, 256, 256, 256, self.vocab_dim + 3]
        self.D_DIMS = [256, 256, 256, 256, 256, 256]
        self.max_num_visit = max_num_visit
        self.max_code_visit = 101


class PointWiseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(PointWiseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.bias = self.add_variable("bias", shape=[self.num_outputs])

    def call(self, x, y):
        return x * y + self.bias


def prob2onehot(prob):
    return tf.cast((tf.reduce_max(prob, axis=-1, keepdims=True) - prob) == 0, tf.float32)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim) for dim in config.G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5, center=False, scale=False)] + \
                                 [tf.keras.layers.BatchNormalization(epsilon=1e-5) for _ in config.G_DIMS[1:-1]]
        self.output_layer_code = tf.keras.layers.Dense(config.G_DIMS[-1], activation=tf.nn.sigmoid)
        self.output_layer_stay = tf.keras.layers.Dense(8, activation=tf.nn.softmax)
        self.output_layer_left = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        self.output_layer_right = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        self.condition_layer = tf.keras.layers.Dense(config.G_DIMS[0])
        self.pointwiselayer = PointWiseLayer(config.G_DIMS[0])

    def call(self, condition):
        x = tf.random.normal(shape=[condition.shape.as_list()[0], config.Z_DIM])
        h = self.dense_layers[0](x)
        x = tf.nn.relu(
            self.pointwiselayer(self.batch_norm_layers[0](h, training=False), self.condition_layer(condition)))
        for i in range(1, len(config.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=False))
            x += h
        x = tf.concat((tf.math.round(self.output_layer_code(x)),
                       prob2onehot(self.output_layer_left(x)),
                       prob2onehot(self.output_layer_right(x)),
                       prob2onehot(self.output_layer_stay(x))), axis=-1)
        return x


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        lstm = DropConnectLSTM
        self.layer = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(config.n_layer)]

    def call(self, x):
        for i in range(config.n_layer):
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

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
                self.recurrent_kernel[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:],
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
        code = tf.reshape(self.encoding_feature(code), [others.shape[0], max_num_visit, num_code+1])

        target = tf.gather(1 - self.encoding_target(target),weight_idx,axis=1)
        target = tf.tile(tf.expand_dims(target,-2),[1,max_num_visit,1])
        code = tf.concat((code, target),axis=-1)
        interval = others[:,:,27:37]
        interval_backward = tf.concat((tf.zeros((others.shape[0],1,10), dtype=tf.float32),
                                       tf.reverse_sequence(interval, length, seq_axis=1)[:,:-1]),axis=1)

        feature = tf.concat((code, others[:,:,:27], others[:,:,37:]),axis=-1)
        feature_backward = tf.reverse_sequence(feature, length, seq_axis=1)

        x_forward = self.linear_forward(tf.concat((feature, interval),axis=-1)[:,:-2])
        x_backward = self.linear_backward(tf.concat((feature_backward, interval_backward),axis=-1)[:,:-2])
        return x_forward, x_backward


class FeatureNet(tf.keras.Model):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.embeddings = Embedding()
        self.lstm_forward = LSTM()
        self.lstm_backward = LSTM()
        self.dense = tf.keras.layers.Dense(256)
        self.mlp0 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(256)

    def call(self, code, others, length, target, idx):
        length = tf.squeeze(length)
        x_forward, x_backward = self.embeddings(code, others, length, target)
        x_forward = self.lstm_forward(x_forward)

        x_backward = tf.reverse_sequence(self.lstm_backward(x_backward), length - 2, seq_axis=1)

        x = tf.concat((x_forward, x_backward), axis=-1)
        x = tf.gather_nd(x, tf.concat(
            (tf.expand_dims(tf.range(code.shape.as_list()[0], dtype=tf.int32), -1), tf.expand_dims(idx, -1)),
            axis=-1))
        feature_vec = self.mlp1(self.mlp0(x))
        feature_vec = feature_vec / tf.math.sqrt(tf.reduce_sum(feature_vec ** 2, axis=-1, keepdims=True))
        return feature_vec


def gen_dataset(k):
    model = FeatureNet()
    generator = Generator()

    checkpoint_directory_model = "training_checkpoints_bilstm_10"
    checkpoint_prefix_model = os.path.join(checkpoint_directory_model, "ckpt")

    checkpoint_directory_generator = "training_checkpoints_gan"
    checkpoint_prefix_generator = os.path.join(checkpoint_directory_generator, "ckpt")

    checkpoint_model = tf.train.Checkpoint(model=model)
    checkpoint_model.restore(checkpoint_prefix_model + '-20').expect_partial()

    checkpoint_generator = tf.train.Checkpoint(generator=generator)
    checkpoint_generator.restore(checkpoint_prefix_generator + '-30').expect_partial()

    @tf.function
    def insert_step(batch_code, batch_others, batch_length, batch_target, batch_idx):
        batch_latent = model(batch_code, batch_others, batch_length, batch_target, batch_idx)
        batch_latent = tf.tile(tf.squeeze(batch_latent), [10, 1])
        batch_insert = generator(batch_latent)
        return batch_insert

    def gen_batch(batch_code, batch_others, batch_length, batch_target):
        tmp_target = []
        for i in batch_target:
            tmp = np.zeros(263)
            tmp[i[i >= 1]] = 1
            tmp_target.append(tmp)
        tmp_target = np.array(tmp_target)
        max_length = np.max(batch_length)
        for _ in range(1):
            insert_sequence = []
            for _ in range(len(batch_code)):
                tmp_sequence = np.arange(1,max_length-1)
                np.random.shuffle(tmp_sequence)
                insert_sequence.append(tmp_sequence)
            insert_sequence = np.array(insert_sequence)
            for i in range(max_length-2):
                batch_idx = insert_sequence[:, i]
                update_idx = np.arange(len(batch_code))[batch_idx - batch_length < 1]
                batch_insert = insert_step(tf.convert_to_tensor(batch_code, dtype=tf.int32),
                                           tf.convert_to_tensor(batch_others, dtype=tf.float32),
                                           tf.convert_to_tensor(batch_length, dtype=tf.int32),
                                           tf.convert_to_tensor(batch_target, dtype=tf.int32),
                                           tf.convert_to_tensor(batch_idx, dtype=tf.int32)).numpy()

                correct_output = batch_insert[:len(batch_code)]
                for attempt in range(10):
                    target_violate = np.sum((tmp_target[:,1:]+correct_output[:,245:507])==2,axis=-1)>0
                    num_code_visit = np.sum(correct_output[:, 1:507] == 1, axis=-1)
                    left_edge = correct_output[:, 509] == 1
                    right_edge = correct_output[:, 519] == 1
                    violate = np.logical_or(np.logical_or(num_code_visit == 0, np.logical_or(left_edge, right_edge)),target_violate)
                    if np.sum(violate[update_idx]) == 0:
                        break
                    correct_output[violate] = batch_insert[attempt * len(batch_code):(attempt + 1) * len(batch_code)][violate]

                insert_code = [np.arange(506)[correct_output[n, 1:507] == 1] for n in range(len(batch_code))]
                insert_code = np.array(
                    [np.pad(w, (0, config.max_code_visit - len(w)), 'constant', constant_values=-1) for w in
                     insert_code])
                batch_code[update_idx, batch_idx[update_idx]] = insert_code[update_idx] + 1
                batch_others[update_idx, batch_idx[update_idx], 19:27] = correct_output[update_idx, 529:537]
                batch_others[update_idx, batch_idx[update_idx], 27:37] = correct_output[update_idx, 509:519]
                batch_others[update_idx, batch_idx[update_idx], -2:] = correct_output[update_idx, 507:509]
                right_update_idx = batch_idx[update_idx] + 1
                batch_others[update_idx, right_update_idx, 27:37] = correct_output[update_idx, 519:529]

        tmp_code = [w[:v] for w, v in zip(batch_code, batch_length)]
        tmp_others = [w[:v] for w, v in zip(batch_others, batch_length)]
        tmp_history = []
        for w,v in zip(batch_code,batch_target):
            tmp = np.unique(w)-244
            tmp = tmp[tmp >= 1]
            tmp_history.append(np.concatenate([tmp, np.zeros(200-len(tmp))],axis=-1))
        return tmp_code, tmp_others, tmp_history
    n_batch = int(len(length_train) / config.batchsize)
    feature_tmp = []
    summary_tmp = []
    history_tmp = []
    for j in range(n_batch):
        x, y, z = gen_batch(feature_train[j * config.batchsize:(j + 1) * config.batchsize],
                         summary_train[j * config.batchsize:(j + 1) * config.batchsize],
                         length_train[j * config.batchsize:(j + 1) * config.batchsize],
                         target_train[j * config.batchsize:(j + 1) * config.batchsize])
        feature_tmp.extend(x)
        summary_tmp.extend(y)
        history_tmp.extend(z)
    x, y, z = gen_batch(feature_train[n_batch * config.batchsize:],
                     summary_train[n_batch * config.batchsize:],
                     length_train[n_batch * config.batchsize:],
                     target_train[n_batch * config.batchsize:])
    feature_tmp.extend(x)
    summary_tmp.extend(y)
    history_tmp.extend(z)
    with open('syn/code_train_' + str(k) + '.pkl', 'wb') as f:
        pickle.dump(feature_tmp, f)
    with open('syn/others_train_' + str(k) + '.pkl', 'wb') as f:
        pickle.dump(summary_tmp, f)
    np.save('history_train_'+str(k),history_tmp)


if __name__ == '__main__':
    config = Config()
    lengths = np.load('../XINMENG/length.npy').astype('int32')
    features = np.load('../XINMENG/code.npy').astype('int32') + 1
    summaries = np.load('../XINMENG/others.npy').astype('float32')
    targets = np.load('../XINMENG/target.npy').astype('int32') + 1

    train_idx = np.load('train_idx_10.npy')
    weight_idx = tf.constant(np.arange(283)[np.load('../AUG2/label_weight.npy') >= 50], dtype=tf.int32)

    lengths = lengths[train_idx]
    features = features[train_idx]
    summaries = summaries[train_idx]
    targets = targets[train_idx]

    order = np.argsort(lengths)
    length_train = lengths[order]
    feature_train = features[order]
    summary_train = summaries[order]
    target_train = targets[order]
    for a in range(10):
        t = time.time()
        gen_dataset(a)
        print(time.time() - t)
