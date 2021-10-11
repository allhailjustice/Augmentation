import tensorflow as tf
import numpy as np
import time
import os
import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.config.experimental.set_memory_growth = True
checkpoint_directory = "training_checkpoints_gan"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

num_code = 506 + 3
batchsize = 2000

Z_DIM = 128
G_DIMS = [256, 256, 256, 256, 256, 256, num_code]
D_DIMS = [256, 256, 256, 256, 256, 256]


class PointWiseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(PointWiseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.bias = self.add_variable("bias",
                                      shape=[self.num_outputs])

    def call(self, x, y):
        return x * y + self.bias


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim) for dim in G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5,center=False, scale=False)] + \
                                 [tf.keras.layers.BatchNormalization(epsilon=1e-5) for _ in G_DIMS[1:-1]]
        self.output_layer_code = tf.keras.layers.Dense(G_DIMS[-1], activation=tf.nn.sigmoid)
        self.output_layer_stay = tf.keras.layers.Dense(8, activation=tf.nn.softmax)
        self.output_layer_left = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        self.output_layer_right = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        self.condition_layer = tf.keras.layers.Dense(G_DIMS[0])
        self.pointwiselayer = PointWiseLayer(G_DIMS[0])

    def call(self, x, condition, training):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(
            self.pointwiselayer(self.batch_norm_layers[0](h, training=training), self.condition_layer(condition)))
        for i in range(1,len(G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=training))
            x += h
        x = tf.concat((self.output_layer_code(x),self.output_layer_left(x),
                       self.output_layer_right(x),self.output_layer_stay(x)),axis=-1)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu)
                             for dim in D_DIMS]
        self.layer_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5,center=False, scale=False)] + \
                                 [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in D_DIMS[1:]]
        self.output_layer = tf.keras.layers.Dense(1)
        self.condition_layer = tf.keras.layers.Dense(D_DIMS[0])
        self.pointwiselayer = PointWiseLayer(D_DIMS[0])

    def call(self, x, condition):
        x = self.dense_layers[0](x)
        x = self.pointwiselayer(self.layer_norm_layers[0](x), self.condition_layer(condition))
        for i in range(1,len(D_DIMS)):
            h = self.dense_layers[i](x)
            h = self.layer_norm_layers[i](h)
            x += h
        x = self.output_layer(x)
        return x


def train():
    feature_description = {
        'word': tf.io.FixedLenFeature([537], tf.float32),
        'condition': tf.io.FixedLenFeature([256], tf.float32)
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed['word'], parsed['condition']

    dataset_train = tf.data.TFRecordDataset('bilstm_10.tfrecord')
    parsed_dataset_train = dataset_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    parsed_dataset_train = parsed_dataset_train.batch(batchsize, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    generator = Generator()
    discriminator = Discriminator()

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator,
                                     discriminator_optimizer=discriminator_optimizer, discriminator=discriminator)
    # checkpoint.restore(checkpoint_prefix+'-40')

    @tf.function
    def d_step(word, condition):
        real = word
        z = tf.random.normal(shape=[batchsize, Z_DIM])

        epsilon = tf.random.uniform(
            shape=[batchsize, 1],
            minval=0.,
            maxval=1.)

        with tf.GradientTape() as disc_tape:
            synthetic = generator(z, condition, False)
            interpolate = real + epsilon * (synthetic - real)

            real_output = discriminator(real, condition)
            fake_output = discriminator(synthetic, condition)

            w_distance = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output))
            with tf.GradientTape() as t:
                t.watch([interpolate, condition])
                interpolate_output = discriminator(interpolate, condition)
            w_grad = t.gradient(interpolate_output, [interpolate, condition])
            slopes = tf.sqrt(tf.reduce_sum(tf.square(w_grad[0]), 1)+tf.reduce_sum(tf.square(w_grad[1]), 1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            disc_loss = 10 * gradient_penalty + w_distance

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, w_distance

    @tf.function
    def g_step(condition):
        z = tf.random.normal(shape=[batchsize, Z_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = generator(z, condition, True)

            fake_output = discriminator(synthetic, condition)

            gen_loss = -tf.reduce_mean(fake_output) + tf.reduce_sum(generator.losses)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    @tf.function
    def train_step(batch):
        word, condition = batch
        # word = tf.cast(word, tf.float32)
        disc_loss, w_distance = d_step(word, condition)
        g_step(condition)
        return disc_loss, w_distance

    print('training start')
    for epoch in range(3000):
        start_time = time.time()
        total_loss = 0.0
        total_w = 0.0
        step = 0.0
        for args in parsed_dataset_train:
            loss, w = train_step(args)
            total_loss += loss
            total_w += w
            step += 1
        duration_epoch = time.time() - start_time
        format_str = 'epoch: %d, loss = %f, w = %f, (%.2f)'
        if epoch % 10 == 9:
            print(format_str % (epoch, -total_loss / step, -total_w / step, duration_epoch))
        if epoch % 50 == 49:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    train()
