import os
from config import Config
from AE import AE
from AE_data_loader import load
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_history(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()


def plot_true_and_prediction(trained_model, true_data):
    if true_data.shape[-1] == 1:
        prediction = trained_model(true_data)

        shape = true_data.shape
        samples = shape[0] * shape[1]
        full_ground_truth = tf.reshape(true_data, [samples, *shape[2:]])
        full_prediction = tf.reshape(prediction, [samples, *shape[2:]])

        start = int(tf.random.uniform([1], minval=0, maxval=samples-280-1))
        end = start + 280
        plt.plot(full_ground_truth[start:end], label="Ground Truth")
        plt.plot(full_prediction[start:end], label="Prediction")
        plt.legend()
        plt.show()


def test_model(trained_model, data):
    prediction = trained_model(data)
    error = tf.losses.MeanSquaredError()(prediction, data)
    print(f"Error on test data is: {error}")


if __name__ == "__main__":
    config = Config()
    model = AE(config)

    train_data, valid_data, test_data, unshuffled_full_data = load(config)

    input_shape = train_data.shape[1:]

    model.my_build(input_shape)
    model.compile(optimizer=config.optimizer, run_eagerly=True, loss="mse")

    history_dict = model.fit(x=train_data,
                             y=train_data,
                             epochs=config.num_epochs,
                             validation_data=(valid_data, valid_data),
                             callbacks=[
                                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
                             ],
                             )

    plot_history(history_dict)
    test_model(model, test_data)
    plot_true_and_prediction(model, unshuffled_full_data)
