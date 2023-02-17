import numpy as np
import tensorflow as tf

# Workaround needed for JetBrains IDE
# https://youtrack.jetbrains.com/issue/PY-53599/tensorflow.keras-subpackages-are-unresolved
keras = tf.keras
from keras.optimizers import Adam


class Config:
    def __init__(self):
        # Define dataset
        # "taxi" and "wind" are possible values
        self.dataset = "taxi"
        self.num_epochs = 500

        self.optimizer = Adam()

        """ Model parameters, you can leave them as they are """

        if self.dataset == "taxi":
            # Define Model parameters
            self.encoder = {"Conv_1": [4, 3, 1],
                            "Conv_2": [8, 3, 1],
                            "Conv_3": [16, 3, 1],
                            "Dense_encoder": 128}

            self.latent_dim = 2

            # Last layer must be treated carefully
            self.decoder = {"Dense_decoder": 128,
                            "DeConv_3": [8, 3, 1],
                            "DeConv_2": [4, 3, 1],
                            "last_layer": [1, 3, 1]}

            # Number of data points in one sequence
            self.period_steps = 48

        elif self.dataset == "wind":
            # Define Model parameters
            self.encoder = {"Conv_1": [16, 3, 1],
                            "Conv_2": [32, 3, 1],
                            "Conv_3": [64, 3, 1],
                            "Dense_encoder": 256}

            self.latent_dim = 10

            # Last layer must be treated carefully
            self.decoder = {"Dense_decoder": 256,
                            "DeConv_3": [32, 3, 1],
                            "DeConv_2": [16, 3, 1],
                            "last_layer": [8, 3, 1]}

            # Number of data points in one sequence
            self.period_steps = 28
