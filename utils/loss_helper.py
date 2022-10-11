import tensorflow as tf
import tensorflow_probability.python.distributions as tfp
from tensorflow import keras


def loss_helper(encoder, decoder, stochastic_layer_count, objective):
    def loss(x_true, x_pred):
        if stochastic_layer_count == 1:
            z_mean, z_sigma, z = encoder(x_true)
        else:
            _, z_mean, z_sigma, z = encoder(x_true)

        out = decoder(z)

        z = tf.reshape(z, (5, -1, 50))
        out = tf.reshape(out, (5, -1, 784))

        p_z = tf.reduce_sum(tfp.Normal(0, 1).log_prob(z), axis=-1)

        t = tfp.Bernoulli(probs=out, dtype=tf.float32).log_prob(x_true)  # broadcast

        p_x_z = tf.reduce_sum(t, axis=-1)

        q_z_x = tf.reduce_sum(tfp.Normal(z_mean, z_sigma + 1e-6).log_prob(z), axis=-1)

        log_w = p_z + p_x_z - q_z_x
        # print(p_x_z)

        if objective == "vae":
            # First the mean over the samples
            # Second the mean over the elbos
            mean_loss = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
            # print(mean_loss)

        elif objective == "iwae":
            w = tf.exp(log_w - tf.reduce_max(log_w, axis=0, keepdims=True))
            w_norm = w / tf.reduce_sum(w, axis=0, keepdims=True)
            w_norm = tf.stop_gradient(w_norm)
            mean_loss = tf.reduce_mean(tf.reduce_sum(w_norm * log_w, axis=0))

        else:
            raise Exception("Not defined objective!")

        return -mean_loss

    return loss
