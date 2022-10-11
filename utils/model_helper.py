from tensorflow import keras
from tensorflow.keras import layers

from utils.sampling_helper import Sampling_Normal, k

latent_dim_50 = 50
latent_dim_100 = 100

det_size_200 = 200
det_size_100 = 100


def create_model(stochastic_layer_count, image_size, implementation):
    if stochastic_layer_count == 1:

        # Encoder
        encoder_inputs = keras.Input(shape=image_size ** 2)
        encoder_1 = layers.Dense(det_size_200, activation="tanh")(encoder_inputs)
        encoder_1 = layers.Dense(det_size_200, activation="tanh")(encoder_1)
        z_mean_1 = layers.Dense(latent_dim_50, name="z_mean_1", activation="linear")(encoder_1)
        z_sigma_1 = layers.Dense(latent_dim_50, name="z_sigma_1", activation="exponential")(encoder_1)
        z_1 = Sampling_Normal()([z_mean_1, z_sigma_1])
        encoder = keras.Model(encoder_inputs, [z_mean_1, z_sigma_1, z_1], name="encoder")
        encoder.summary()

        # Decoder
        latent_inputs = keras.Input(shape=(k, latent_dim_50))
        decoder_1 = layers.Dense(det_size_200, activation="tanh")(latent_inputs)
        decoder_1 = layers.Dense(det_size_200, activation="tanh")(decoder_1)
        decoder_outputs = layers.Dense(image_size ** 2, activation="sigmoid")(decoder_1)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

    elif stochastic_layer_count == 2:

        # Encoder
        encoder_inputs = keras.Input(shape=image_size ** 2)
        encoder_1 = layers.Dense(det_size_200, activation="tanh")(encoder_inputs)
        encoder_1 = layers.Dense(det_size_200, activation="tanh")(encoder_1)
        encoder_z_mean_1 = layers.Dense(latent_dim_100, name="encoder_z_mean_1", activation="linear")(encoder_1)
        encoder_z_sigma_1 = layers.Dense(latent_dim_100, name="encoder_z_sigma_1", activation="exponential")(encoder_1)
        encoder_z_1 = Sampling_Normal()([encoder_z_mean_1, encoder_z_sigma_1])

        encoder_2 = layers.Dense(det_size_100, activation="tanh")(encoder_z_1)
        encoder_2 = layers.Dense(det_size_100, activation="tanh")(encoder_2)
        encoder_z_mean_2 = layers.Dense(latent_dim_50, name="encoder_z_mean_2", activation="linear")(encoder_2)
        encoder_z_sigma_2 = layers.Dense(latent_dim_50, name="encoder_z_sigma_2", activation="exponential")(encoder_2)
        encoder_z_2 = Sampling_Normal()([encoder_z_mean_2, encoder_z_sigma_2])
        encoder = keras.Model(encoder_inputs, [encoder_z_mean_1, encoder_z_mean_2, encoder_z_sigma_2, encoder_z_2],
                              name="encoder")
        encoder.summary()

        # Decoder
        latent_inputs = keras.Input(shape=(latent_dim_50,))
        decoder_2 = layers.Dense(det_size_100, activation="tanh")(latent_inputs)
        decoder_2 = layers.Dense(det_size_100, activation="tanh")(decoder_2)
        decoder_z_mean_2 = layers.Dense(latent_dim_100, name="decoder_z_mean_2", activation="linear")(decoder_2)
        decoder_z_sigma_2 = layers.Dense(latent_dim_100, name="decoder_z_sigma_2", activation="exponential")(decoder_2)
        decoder_z_2 = Sampling_Normal()([decoder_z_mean_2, decoder_z_sigma_2])

        decoder_1 = layers.Dense(det_size_200, activation="tanh")(decoder_z_2)
        decoder_1 = layers.Dense(det_size_200, activation="tanh")(decoder_1)
        decoder_outputs = layers.Dense(image_size ** 2, activation="sigmoid")(decoder_1)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

    else:
        raise Exception("Not defined stochastic layer count!")

    outputs = decoder(encoder(encoder_inputs)[-1])
    vae = keras.Model(encoder_inputs, outputs, name='vae')

    return vae, encoder, decoder
