import numpy as np
from tensorflow.keras.optimizers import Adam


class Config:
    def __init__(self):
        # Define mode and dataset
        self.mode = False
        self.dataset = "zeros"
        self.num_epochs = 5

        # Define Model parameters
        # Priors: "Gaussian",
        self.prior = "Gaussian"
        self.optimizer = Adam()
        self.encoder = {"Conv": [5, 3, 2],
                        "Dense": 20}
        self.latent_dim = 1

        # Last layer must be treated carefully
        self.decoder = {"Dense": 20,
                        "last_layer": [3, 3, 2]}
