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
        self.dataset = "zeros"
        self.num_epochs = 5

        # Define Model parameters
        self.optimizer = Adam()
        self.encoder = {"Conv": [5, 7, 2],
                        "Dense": 20}
        self.latent_dim = 1

        # Last layer must be treated carefully
        self.decoder = {"Dense": 20,
                        "last_layer": [1, 7, 2]}
