import os
from config import Config
from AE import AE
from AE_data_loader import load
import tensorflow as tf


if __name__ == "__main__":
    config = Config()
    model = AE(config)

    train_data = load(config)

    input_shape = train_data.element_spec.shape[1:]

    model.my_build(input_shape)
    model.compile(optimizer=config.optimizer, run_eagerly=True)

    model.fit(x=train_data, epochs=100)
