import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# tf.config.experimental.set_memory_growth = True
checkpoint_directory = "training_checkpoints_bilstm_10"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
num_code = 526
batchsize = 10
max_num_visit = 200


def _array_float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _array_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))


def serialize_example(word, condition):
    feature = {
        'word': _array_float_feature(word),
        'condition': _array_float_feature(condition)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


class Config(object):
    def __init__(self):
        self.vocab_dim = num_code
        self.lstm_dim = 256
        self.n_layer = 3
        self.batchsize = batchsize


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
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self,is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units,self.units*4]),0.0)
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

    def call(self, code, others, length):
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

        interval = others[:,:,27:37]
        interval_backward = tf.concat((tf.zeros((code.shape[0],1,10), dtype=tf.float32),
                                       tf.reverse_sequence(interval, length, seq_axis=1)[:,:-1]),axis=1)

        feature = tf.concat((code, others[:,:,:27], others[:,:,37:]),axis=-1)
        feature_backward = tf.reverse_sequence(feature, length, seq_axis=1)

        x_forward = self.linear_forward(tf.concat((feature, interval),axis=-1)[:,:-2])
        x_backward = self.linear_backward(tf.concat((feature_backward, interval_backward),axis=-1)[:,:-2])
        return x_forward, x_backward, x


class FeatureNet(tf.keras.Model):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.embeddings = Embedding()
        self.lstm_forward = LSTM()
        self.lstm_backward = LSTM()
        self.dense = tf.keras.layers.Dense(256)
        self.mlp0 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(256)

    def call(self, code, others, length, is_training=False):
        length = tf.squeeze(length)
        x_forward, x_backward, vec = self.embeddings(code, others, length)
        x_forward = self.lstm_forward(x_forward, is_training)

        x_backward = tf.reverse_sequence(self.lstm_backward(x_backward, is_training),length - 2, seq_axis=1)

        x_forward = tf.boolean_mask(x_forward, tf.sequence_mask(length - 2, max_num_visit-2))
        x_backward = tf.boolean_mask(x_backward, tf.sequence_mask(length - 2, max_num_visit - 2))
        x = tf.concat((x_forward, x_backward), axis=-1)
        feature_vec = self.mlp1(self.mlp0(x))
        feature_vec = feature_vec / tf.math.sqrt(tf.reduce_sum(feature_vec ** 2, axis=-1, keepdims=True))
        return feature_vec, vec



def train():
    lengths = np.load('length.npy')
    features = np.load('code.npy') + 1
    summaries = np.load('others.npy')

    train_idx = np.load('train_idx_10.npy')
    order = np.arange(len(train_idx))
    length_train = lengths[train_idx][order]
    feature_train = features[train_idx][order]
    summary_train = summaries[train_idx][order]

    dataset_train = tf.data.Dataset.from_tensor_slices((feature_train.astype('int32'),
                                                        summary_train.astype('float32'),
                                                        length_train.astype('int32')))
    parsed_dataset_train = dataset_train.batch(config.batchsize, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    model = FeatureNet()
    checkpoint = tf.train.Checkpoint(model=model)
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    checkpoint.restore(checkpoint_prefix + '-15').expect_partial()
    print('start')
    @tf.function
    def one_step(code_o, others_o, length_o, target_o):
        return model(code_o, others_o,length_o,target_o, is_training=False)

    with tf.io.TFRecordWriter('bilstm_10.tfrecord') as writer:
        for i,batch in enumerate(parsed_dataset_train):
            code, others, length, target = batch
            length = tf.squeeze(length)
            condition_vector, vec = one_step(code, others, length, target)
            for x,y in zip(condition_vector.numpy(), vec.numpy()):
                example = serialize_example(y,x)
                writer.write(example)
            if i % 500 == 499:
                print(int(i/500))


if __name__ == '__main__':
    config = Config()
    train()