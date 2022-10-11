import tensorflow as tf
import tensorflow_probability.python.distributions as tfp

from tensorflow.keras import layers

k = 5  # 1, 5, 50


class Sampling_Normal(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_sigma = inputs
        z = tfp.Normal(loc=z_mean, scale=z_sigma + 1e-6).sample(k)
        z = tf.reshape(z, (-1, k, 50))
        return z
