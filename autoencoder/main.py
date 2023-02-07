import os
from config import Config
from AE import AE
from AE_data_loader import load
import tensorflow as tf
from matplotlib import pyplot as plt

if __name__ == "__main__":
    config = Config()
    model = AE(config)

    train_data, valid_data = load(config)

    input_shape = train_data.shape[1:]

    model.my_build(input_shape)
    model.compile(optimizer=config.optimizer, run_eagerly=True, loss="mse")

    history = model.fit(x=train_data,
                        y=train_data,
                        epochs=1000,
                        batch_size=100,
                        validation_data=(valid_data, valid_data),
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
                        ],
                        )
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
