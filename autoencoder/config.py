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
        self.encoder = {"Conv_1": [8, 7, 1],
                        "Conv_2": [16, 5, 1],
                        "Dense": 512}

        self.latent_dim = 20

        # Last layer must be treated carefully
        self.decoder = {"Dense": 512,
                        "DeConv": [8, 5, 1],
                        "last_layer": [1, 7, 1]}
