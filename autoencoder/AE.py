import tensorflow as tf

# Workaround needed for JetBrains IDE
# https://youtrack.jetbrains.com/issue/PY-53599/tensorflow.keras-subpackages-are-unresolved
keras = tf.keras
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Flatten,
    LeakyReLU,
    Activation,
    Conv1D,
    Reshape,
    Conv1DTranspose,
    Dropout,
    GaussianNoise)
import numpy as np


class AE(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = config.num_epochs

        self.latent_dim = config.latent_dim

        self.encoder_config = config.encoder
        self.encoder = Model()

        self.decoder_config = config.decoder
        self.decoder = Model()

        self.loss_func = tf.keras.losses.MeanSquaredError()

    def build_encoder(self, input_shape):
        """THE ENCODER"""
        encoder_input = Input(shape=input_shape, name="encoder_input")
        x = GaussianNoise(stddev=0.03)(encoder_input)
        # x = encoder_input

        layer_counter = 0

        # Iterate over conv layers in encoder
        for layer in self.encoder_config:
            if "Conv" in layer:
                filters, kernel_size, stride = self.encoder_config[layer]
                x = Conv1D(filters=filters,
                           kernel_size=kernel_size,
                           strides=stride,
                           padding="same",
                           name=f"{layer_counter}_encoder_conv")(x)
                x = LeakyReLU()(x)
                x = Dropout(.2)(x)

                layer_counter += 1

        # Make sure we remember shape
        # And flatten before any dense layers
        shape_before_flatten = x.shape[1:]
        x = Flatten()(x)

        # Iterate over dense layers in encoder
        for layer in self.encoder_config:
            if "Dense" in layer:
                num_neurons = self.encoder_config[layer]
                x = Dense(num_neurons, name=f"{layer_counter}_encoder_dense")(x)
                x = LeakyReLU()(x)
                x = Dropout(.2)(x)

                layer_counter += 1

        num_neurons = self.latent_dim
        x = Dense(num_neurons, name="latent_layer")(x)

        # Encoder Model
        self.encoder = Model(
            encoder_input, x, name="encoder"
        )

        return shape_before_flatten

    def build_decoder(self, shape_before_flatten):
        """THE DECODER"""
        decoder_input = Input(shape=self.latent_dim, name="decoder_input")
        x = decoder_input

        layer_counter = 0

        # Iterate over dense layer in decoder
        for layer in self.decoder_config:
            if "Dense" in layer:
                num_neurons = self.decoder_config[layer]
                x = Dense(num_neurons, name=f"{layer_counter}_decoder_dense")(x)
                x = LeakyReLU()(x)
                x = Dropout(.2)(x)

                layer_counter += 1

        # Reshape to shape before dense layers in encoder
        if x.shape[1:].ndims == 1:
            x = Dense(np.prod(shape_before_flatten))(x)
            x = LeakyReLU()(x)
            x = Dropout(.2)(x)

            x = Reshape(shape_before_flatten, name="Reshape_before_conv")(x)

        # Iterate over conv layers in decoder
        for layer in self.decoder_config:
            if "Conv" in layer:
                filters, kernel_size, stride = self.decoder_config[layer]
                x = Conv1DTranspose(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding="same",
                                    name=f"{layer_counter}_decoder_conv")(x)
                x = LeakyReLU()(x)
                x = Dropout(.2)(x)

                layer_counter += 1

        filters, kernel_size, stride = self.decoder_config["last_layer"]
        x = Conv1DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=stride,
                            padding="same",
                            name="last_layer")(x)

        # decoder_output = x
        decoder_output = Activation("sigmoid")(x)

        # Decoder Model
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def my_build(self, input_shape):
        """Build the model
        :param input_shape"""

        shape_before_flatten = self.build_encoder(input_shape)
        self.build_decoder(shape_before_flatten)
        self.build([None, *input_shape])
    """
    def train_step(self, data):
        with tf.GradientTape() as tape:
            latent = self.encode(data)

            reconstruction = self.decode(latent)

            loss = self.compute_loss(data, reconstruction)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def compute_loss(self, true, prediction):
        return self.loss_func(true, prediction)
    """
    def encode(self, inputs):
        """Returns the encoder output, including the relevant from sampling for analysis purposes
        [Batch_size, encoding, ...]
        ... empty for standard autoencoder
        """
        return self.encoder(inputs)

    def decode(self, latent_inputs):
        return self.decoder(latent_inputs)

    def call(self, inputs, training=None, mask=None):
        return self.decode(self.encode(inputs))
