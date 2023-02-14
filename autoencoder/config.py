import numpy as np
import tensorflow as tf

# Workaround needed for JetBrains IDE
# https://youtrack.jetbrains.com/issue/PY-53599/tensorflow.keras-subpackages-are-unresolved
keras = tf.keras
from keras.optimizers import Adam


class Config:
    def __init__(self):
        # Define mode and dataset
        self.mode = False
        self.dataset = "taxi"
        self.num_epochs = 500

        # Define Model parameters
        self.optimizer = Adam()
        self.encoder = {"Conv_1": [4, 3, 1],
                        "Conv_2": [8, 3, 1],
                        "Conv_3": [16, 3, 1],
                        "Dense_encoder": 128}

        self.latent_dim = 1

        # Last layer must be treated carefully
        self.decoder = {"Dense_decoder": 128,
                        "DeConv_3": [8, 3, 1],
                        "DeConv_2": [4, 3, 1],
                        "last_layer": [1, 3, 1]}
