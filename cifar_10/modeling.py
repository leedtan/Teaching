from functools import partial

import numpy as np
import tensorflow as tf

NUM_CLASSES = 10

tfph = tf.compat.v1.placeholder
tf.compat.v1.disable_eager_execution()

# Tensorflow aggressively deprecated static support for layer_norm.
# This was some online code I found for people to get their old layer_norm function.
# I've never used this in tf2, but I assume it should be identical
def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.keras.layers.LayerNormalization(name=name, axis=-1, epsilon=1e-12, dtype=tf.float32)(input_tensor)


class ResidualConv:
    def __init__(self, layer_sizes):
        self.layers = [
            partial(tf.compat.v1.layers.conv2d, filters=size, kernel_size=3, padding="SAME") for size in layer_sizes
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


def softmax(v, axes, mask=None):
    #     softmax over multiple axes (over these axes, they all a probability distribution)
    # handles masking for use in nlp transformers
    if mask is not None:
        v = v * mask
    v_max = tf.reduce_max(v, axes, keepdims=True)
    v_stable = v - v_max
    v_exp = tf.exp(v_stable)
    if mask is not None:
        v_exp = v_exp * mask
    v_exp_sum = tf.reduce_sum(v_exp, axes, keepdims=True)
    return v_exp / v_exp_sum


class Attention:
    # image: https://lilianweng.github.io/posts/2018-06-24-attention/transformer.png
    def __init__(self, size):
        self.size = size

    def forward(self, X, einsum=True, dbg=False, ff_layer=True, create_pos_enc=False):
        # input shape: bs, h, w, ndim
        self.X = X
        # batch, queryy, queryx, keyy, keyx, filters
        if create_pos_enc:
            self.length = self.X.shape[1]
            bs = tf.shape(self.X)[0]
            self.posy = tf.reshape(
                tf.range(tf.cast(self.length, tf.float32), dtype=tf.float32),
                (1, 1, -1, 1),
            )
            self.posx = tf.reshape(
                tf.range(tf.cast(self.length, tf.float32), dtype=tf.float32),
                (1, -1, 1, 1),
            )
            self.frequency = tf.reshape(tf.range(1, 6, dtype=tf.float32), (1, 1, 1, -1))
            self.input_sin_x = self.posx / self.frequency
            self.input_sin_y = self.posy / self.frequency
            self.pos_sin_x = tf.tile(tf.sin(self.input_sin_x), (bs, 1, self.length, 1))
            self.pos_cos_x = tf.tile(tf.cos(self.input_sin_x), (bs, 1, self.length, 1))
            self.pos_sin_y = tf.tile(tf.sin(self.input_sin_y), (bs, self.length, 1, 1))
            self.pos_cos_y = tf.tile(tf.cos(self.input_sin_y), (bs, self.length, 1, 1))
            input_signal = tf.concat(
                (
                    self.X,
                    self.pos_sin_x,
                    self.pos_cos_x,
                    self.pos_sin_y,
                    self.pos_cos_y,
                ),
                axis=-1,
            )
        else:
            input_signal = X
        self.q, self.k, self.v = [
            tf.tanh(tf.compat.v1.layers.conv2d(input_signal, filters=self.size, kernel_size=3, padding="SAME"))
            for _ in range(3)
        ]
        if not einsum or dbg:
            self.q_expanded = tf.expand_dims(tf.expand_dims(self.q, 3), 3)
            self.k_expanded = tf.expand_dims(tf.expand_dims(self.k, 1), 1)
            self.v_expanded = tf.expand_dims(tf.expand_dims(self.v, 1), 1)
        if einsum:
            self.scale = tf.einsum("bhwz,bijz->bhwij", self.q, self.k)
            # for testing comparison
            if dbg:
                self.scale2 = tf.reduce_sum(self.q_expanded * self.k_expanded, -1)
        else:
            self.scale = tf.reduce_sum(self.q_expanded * self.k_expanded, -1)
        self.soft = softmax(self.scale, axes=[3, 4])
        # shape: batch, queryy, queryx, keyy, keyx, filters
        if einsum:
            # batch, y, x,
            self.a_compressed = tf.einsum("bhwij,bijz->bhwz", self.soft, self.v)
            if dbg:
                self.a2 = tf.expand_dims(self.soft, -1) * self.v_expanded
                self.a_compressed2 = tf.reduce_sum(self.a2, [3, 4])
        else:
            self.a = tf.expand_dims(self.soft, -1) * self.v_expanded
            self.a_compressed = tf.reduce_sum(self.a, [3, 4])
        self.e = layer_norm(self.a_compressed + X)
        if ff_layer:
            self.output = layer_norm(tf.compat.v1.layers.dense(self.e, self.size) + self.e)
        else:
            self.output = self.e
        return self.output


class MHA:
    # image: https://lilianweng.github.io/posts/2018-06-24-attention/transformer.png
    def __init__(self, size):
        self.size = size

    def forward(self, X, heads=13, einsum=True, dbg=True, ff_layer=True, create_pos_enc=False):
        # input shape: bs, h, w, ndim
        self.X = X
        # batch, queryy, queryx, keyy, keyx, heads, filters
        self.length = self.X.shape[1]
        bs = tf.shape(self.X)[0]
        if create_pos_enc:
            self.posy = tf.reshape(
                tf.range(tf.cast(self.length, tf.float32), dtype=tf.float32),
                (1, 1, -1, 1),
            )
            self.posx = tf.reshape(
                tf.range(tf.cast(self.length, tf.float32), dtype=tf.float32),
                (1, -1, 1, 1),
            )
            self.frequency = tf.reshape(tf.range(1, 6, dtype=tf.float32), (1, 1, 1, -1))
            self.input_sin_x = self.posx / self.frequency
            self.input_sin_y = self.posy / self.frequency
            self.pos_sin_x = tf.tile(tf.sin(self.input_sin_x), (bs, 1, self.length, 1))
            self.pos_cos_x = tf.tile(tf.cos(self.input_sin_x), (bs, 1, self.length, 1))
            self.pos_sin_y = tf.tile(tf.sin(self.input_sin_y), (bs, self.length, 1, 1))
            self.pos_cos_y = tf.tile(tf.cos(self.input_sin_y), (bs, self.length, 1, 1))
            input_signal = tf.concat(
                (
                    self.X,
                    self.pos_sin_x,
                    self.pos_cos_x,
                    self.pos_sin_y,
                    self.pos_cos_y,
                ),
                axis=-1,
            )
        else:
            input_signal = X
        # batch, queryy, queryx, keyy, keyx, heads, filters
        self.q, self.k, self.v = [
            tf.tanh(tf.compat.v1.layers.conv2d(input_signal, filters=self.size * heads, kernel_size=3, padding="SAME"))
            for _ in range(3)
        ]
        # batch, y, x, heads, filters
        self.q, self.k, self.v = [tf.reshape(v, [-1, v.shape[1], v.shape[2], heads, self.size]) for v in [self.q, self.k, self.v]]
        if not einsum or dbg:
            self.q_expanded = tf.expand_dims(tf.expand_dims(self.q, 3), 3)
            self.k_expanded = tf.expand_dims(tf.expand_dims(self.k, 1), 1)
            self.v_expanded = tf.expand_dims(tf.expand_dims(self.v, 1), 1)
        if einsum:
            self.scale = tf.einsum("bhwaz,bijaz->bhwija", self.q, self.k)
            # for testing comparison
            if dbg:
                self.scale2 = tf.reduce_sum(self.q_expanded * self.k_expanded, -1)
        else:
            self.scale = tf.reduce_sum(self.q_expanded * self.k_expanded, -1)
        self.soft = softmax(self.scale, axes=[3, 4])
        # shape: batch, queryy, queryx, keyy, keyx, heads, filters
        if einsum:
            # batch, y, x,
            self.a_compressed = tf.einsum("bhwija,bijaz->bhwaz", self.soft, self.v)
            if dbg:
                self.a2 = tf.expand_dims(self.soft, -1) * self.v_expanded
                self.a_compressed2 = tf.reduce_sum(self.a2, [3, 4])
        else:
            self.a = tf.expand_dims(self.soft, -1) * self.v_expanded
            self.a_compressed = tf.reduce_sum(self.a, [3, 4])
        # a_compressed: batch, queryy, queryx, heads, filters
        self.heads_reshaped = tf.reshape(self.a_compressed, [-1, self.length, self.length, heads * self.size])
        self.MHA_output = tf.compat.v1.layers.dense(self.heads_reshaped, self.size)
        self.e = layer_norm(self.MHA_output + X)
        if ff_layer:
            self.output = layer_norm(tf.compat.v1.layers.dense(self.e, self.size) + self.e)
        else:
            self.output = self.e
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
        self.has_attn = False
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
            if isinstance(layer, Attention) and not self.has_attn:
                features = layer.forward(self.features[-1], create_pos_enc=True)
                self.has_attn = True
            else:
                features = layer.forward(self.features[-1])
            features = tf.nn.leaky_relu(features)
            self.features.append(features)
            if len(layers) // 2 == i:
                pivot_features = features
                for j, extra_layer in enumerate(extra_loss_layers):
                    pivot_feautres = extra_layer.forward(pivot_features)
                    pivot_feautres = tf.nn.leaky_relu(pivot_feautres)
                pivot_feautres = tf.compat.v1.layers.flatten(pivot_feautres)
                extra_pred = tf.compat.v1.layers.dense(pivot_feautres, NUM_CLASSES)
                self.extra_loss = tf.compat.v1.losses.softmax_cross_entropy(self.yph, extra_pred)
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
            self.sample_data(step, batch_size)
            ls, _ = self.sess.run([self.loss, self.opt], self.fd)
            self.losses.append(ls)
            if step % 100 == 0:
                val_loss, forecast = self.sess.run([self.loss, self.yhat], self.val_fd)
                actual_pred = np.argmax(forecast, axis=1)
                acc = (actual_pred == self.y_test.flatten()).mean()
                self.val_acc.append(acc)
                self.val_losses.append(val_loss)
                print(val_loss)

    def sample_data(self, step, batch_size):
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
            norm = np.sqrt(np.sum(np.square(grad_x).reshape(-1, 32 * 32 * 3), axis=1))
            distortion = grad_x / norm[:, None, None, None]
            x_trn = x_sample + distortion

        self.fd = {self.xph: x_trn, self.yph: self.y_train_onehot[samples]}
