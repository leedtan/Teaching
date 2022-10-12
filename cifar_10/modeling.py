from functools import partial

import numpy as np
import tensorflow as tf

NUM_CLASSES = 10

tfph = tf.compat.v1.placeholder
tf.compat.v1.disable_eager_execution()


class ResidualConv:
    def __init__(self, layer_sizes):
        self.layers = [
            partial(
                tf.compat.v1.layers.conv2d, filters=size, kernel_size=3, padding="SAME"
            )
            for size in layer_sizes
        ]

    def forward(self, signal):
        x = signal
        self.features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = tf.nn.leaky_relu(x)
            self.features.append(x)
        self.output = signal + x
        return self.output


class Conv:
    def __init__(self, *args, **kwargs):
        self.layer = partial(tf.compat.v1.layers.conv2d, *args, **kwargs)

    def forward(self, signal):
        return self.layer(signal)


class BaseConv(Conv):
    def __init__(self, size):
        super().__init__(filters=size, kernel_size=3, padding="SAME")


class StridedConv(Conv):
    def __init__(self, size):
        super().__init__(filters=size, kernel_size=5, strides=2, padding="SAME")


class Model:
    def __init__(
        self,
        layers=None,
        extra_loss_layers=[StridedConv(64), StridedConv(128)],
        x_test_norm=None,
        x_train_norm=None,
        y_test_onehot=None,
        y_train_onehot=None,
        y_test=None,
    ):

        self.x_test_norm = x_test_norm
        self.x_train_norm = x_train_norm
        self.y_test_onehot = y_test_onehot
        self.y_train_onehot = y_train_onehot
        self.y_test = y_test

        self.xph = tfph(tf.float32, shape=(None, 32, 32, 3))
        self.yph = tfph(tf.int32, shape=(None, NUM_CLASSES))
        self.layers = layers
        self.features = [self.xph]
        for i, layer in enumerate(layers):
            features = layer.forward(self.features[-1])
            features = tf.nn.leaky_relu(features)
            self.features.append(features)
            if len(layers) // 2 == i:
                pivot_features = features
                for j, extra_layer in enumerate(extra_loss_layers):
                    pivot_feautres = layer.forward(pivot_features)
                    pivot_feautres = tf.nn.leaky_relu(pivot_feautres)
                pivot_feautres = tf.compat.v1.layers.flatten(pivot_feautres)
                extra_pred = tf.compat.v1.layers.dense(pivot_feautres, NUM_CLASSES)
                self.extra_loss = tf.compat.v1.losses.softmax_cross_entropy(
                    self.yph, extra_pred
                )
        features = tf.compat.v1.layers.flatten(features)
        features = tf.compat.v1.layers.dense(features, 64)
        features = tf.nn.leaky_relu(features)
        self.features.append(features)
        output_raw = tf.compat.v1.layers.dense(features, NUM_CLASSES)
        self.features.append(output_raw)
        self.yhat = tf.compat.v1.math.softmax(output_raw, axis=1)
        self.celoss = tf.compat.v1.losses.softmax_cross_entropy(self.yph, output_raw)
        tfvars = tf.compat.v1.trainable_variables()
        self.reg = tf.reduce_sum([tf.reduce_sum(tf.square(var)) for var in tfvars])
        self.loss = self.celoss + self.extra_loss * 0.2 + self.reg * 1e-8
        self.opt = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.val_fd = {self.xph: self.x_test_norm, self.yph: y_test_onehot}
        self.grad_x = tf.gradients(self.loss, self.xph)[0]

    def train(self, steps=10000, batch_size=64):
        for step in range(steps):
            samples = np.random.choice(self.x_train_norm.shape[0], batch_size)
            x_sample = self.x_train_norm[samples]
            if step % 3 == 0:
                noise = np.random.randn(*self.x_train_norm[samples].shape) * 1e-2
                x_trn = x_sample + noise
            elif step % 3 == 1:
                self.fd = {self.xph: x_sample, self.yph: self.y_train_onehot[samples]}
                grad_x = self.sess.run(self.grad_x, self.fd)
                x_trn = x_sample + (grad_x > 0) * 1e-2 - 1e-2 / 2
            else:
                self.fd = {self.xph: x_sample, self.yph: self.y_train_onehot[samples]}
                grad_x = self.sess.run(self.grad_x, self.fd)
                grad_x = np.sqrt(np.abs(grad_x)) * ((grad_x > 0) * 2 - 1)
                norm = np.sqrt(
                    np.sum(np.square(grad_x).reshape(-1, 32 * 32 * 3), axis=1)
                )
                distortion = grad_x / norm[:, None, None, None]
                x_trn = x_sample + distortion

            self.fd = {self.xph: x_trn, self.yph: self.y_train_onehot[samples]}
            ls, _ = self.sess.run([self.loss, self.opt], self.fd)
            self.losses.append(ls)
            if step % 100 == 0:
                val_loss, forecast = self.sess.run([self.loss, self.yhat], self.val_fd)
                actual_pred = np.argmax(forecast, axis=1)
                acc = (actual_pred == self.y_test.flatten()).mean()
                self.val_acc.append(acc)
                self.val_losses.append(val_loss)
                print(val_loss)
