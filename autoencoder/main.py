import os
from config import Config
from VAE import VAE
from data_loader import load
import tensorflow as tf


if __name__ == "__main__":
    config = Config()
    model = VAE(config)

    train_data = load(config)

    model.my_build([32, 32, 3])
    model.build([None, 32, 32, 3])
    model.compile(optimizer=config.optimizer, run_eagerly=True)

    model.fit(x=train_data, epochs=100)
