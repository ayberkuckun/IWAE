import numpy as np
import tensorflow as tf
from tensorflow import keras

from experiments.find_active_units import find_active_units
from utils.dataset_helper import prepare_data
from utils.loss_helper import loss_helper
from utils.lr_helper import lr_scheduler

from utils.model_helper import create_model

dataset = "mnist"  # mnist, omniglot, fashionmnist
dataset_implementation = "github"  # paper, github

architecture_implementation = "github"  # paper, github
objective = "vae"  # vae, iwae

stochastic_layer_count = 1  # 1, 2
epochs = 100  # 3280

# loss and k and 2 objectives and log likelihood
# binarize each epoch

x_train, x_test, image_size = prepare_data(dataset, dataset_implementation)

vae, encoder, decoder = create_model(stochastic_layer_count, image_size, architecture_implementation)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="saved_model_weights/vae_epoch_{epoch:02d}-loss_{loss:.2f}-val_loss_{val_loss:.2f}",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)

optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-4)

vae.compile(optimizer=optimizer, loss=loss_helper(encoder, decoder, stochastic_layer_count, objective), run_eagerly=True)
vae.fit(
    x_train,
    x_train,
    epochs=epochs,
    batch_size=20,
    callbacks=[lr_scheduler, checkpoint],
    validation_data=(x_test, x_test)
)

# checkpoint_path = "saved_model_weights/vae_epoch_10-loss_104.69-val_loss_104.54"
#
# vae.load_weights(checkpoint_path)

np.random.seed(66)
idx_data = np.random.permutation(len(x_test))
data = x_test[idx_data]

scores = vae.evaluate(data[:5000], data[:5000], verbose=1)
print('Test loss: ', scores)

find_active_units(encoder, x_test, stochastic_layer_count)


# class VAE(keras.Model):
#     def __init__(self, encoder, decoder, **kwargs):
#         super(VAE, self).__init__(**kwargs)
#         self.encoder = encoder
#         self.decoder = decoder
#         self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
#         self.reconstruction_loss_tracker = keras.metrics.Mean(
#             name="reconstruction_loss"
#         )
#         self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
#
#     @property
#     def metrics(self):
#         return [
#             self.total_loss_tracker,
#             self.reconstruction_loss_tracker,
#             self.kl_loss_tracker,
#         ]
#
#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             if stochastic_layer_count == 1:
#                 z_mean, z_log_var, z = self.encoder(data)
#             elif stochastic_layer_count == 2:
#                 _, z_mean, z_log_var, z = self.encoder(data)
#
#             reconstruction = self.decoder(z)
#
#             reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction)))
#             kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#             kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#             total_loss = reconstruction_loss + kl_loss
#
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)
#         self.reconstruction_loss_tracker.update_state(reconstruction_loss)
#         self.kl_loss_tracker.update_state(kl_loss)
#
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#             "kl_loss": self.kl_loss_tracker.result(),
#         }
#
# vae = VAE(encoder, decoder)
