import tensorflow as tf
from tensorflow.keras import losses
from abc import ABC, abstractmethod
from tensorflow.keras.layers import (
    Layer,
    Dense)
from tensorflow.keras import backend as K
import numpy as np


class non_VAE:
    def sampling_layer(self, inputs, dims):
        return Dense(dims, name="latent")(inputs)

    def compute_loss(self, data, reconstruction, *args):
        mse = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        return mse(data, reconstruction)

    def __call__(self, inputs):
        return inputs


class Gaussian:
    def __init__(self):
        self.kl_factor = 0.2

    def sampling_layer(self, inputs, dims):
        mu = Dense(dims, name="mu")(inputs)
        log_var = Dense(dims, name="log_var")(inputs)
        return mu, log_var

    def compute_loss(self, data, reconstruction, *args):
        z, z_mean, z_log_var = args[0], args[1][0][0], args[1][0][1]
        cross_ent = tf.square(reconstruction - data)
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=data)
        logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        mse = tf.reduce_mean(logpx_z)

        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, z_mean, z_log_var)

        kl = tf.reduce_mean((logpz - logqz_x))

        return -tf.reduce_mean(-logpx_z + self.kl_factor * (logpz - logqz_x))

    def __call__(self, inputs):
        return Gaussian_sampling(name="sampling")(inputs)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


class Gaussian_sampling(Layer):
    def call(self, *inputs):
        mu, log_var = inputs[0]
        epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
        return mu + K.exp(log_var / 2) * epsilon
