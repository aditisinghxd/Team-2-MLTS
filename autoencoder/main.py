import os
from config import Config
from AE import AE
from AE_data_loader import load
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


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

        start = int(tf.random.uniform([1], minval=0, maxval=samples-336-1))
        end = start + 336
        plt.plot(full_ground_truth[start:end], label="Ground Truth")
        plt.plot(full_prediction[start:end], label="Prediction")
        plt.legend()
        plt.title("Plot for one whole week")
        plt.show()


def set_anomaly_threshold(trained_model, data):
    prediction = trained_model(data)
    loss = tf.reduce_mean(tf.losses.MeanAbsoluteError(reduction="none")(prediction, data), axis=1)
    return np.max(loss)


def find_anomalies(trained_model, data, threshold, dates):
    prediction = trained_model(data)
    loss = tf.reduce_mean(tf.losses.MeanAbsoluteError(reduction="none")(prediction, data), axis=1)

    plt.hist(loss, bins=40)
    plt.axvline(x=threshold, color='r', label='Anomaly threshold')
    plt.xlabel("Reconstruction loss in a week")
    plt.ylabel("Number of samples with loss")
    plt.show()

    anomalies = loss >= threshold
    indices = np.where(anomalies)
    print(f"Number of anomalies: {np.sum(anomalies)}")
    print(f"Day where anomaly occurred: {indices}")

    shape = data.shape
    samples = shape[0] * shape[1]
    full_ground_truth = tf.reshape(data, [samples, *shape[2:]])
    for index in indices[0]:

        if (index+3)*48 > samples:
            start = (index - 7) * 48
            end = index * 48
        elif index <= 3:
            start = 0
            end = 7 * 48
        else:
            start = (index - 3) * 48
            end = (index+4)*48

        time_values = np.arange(start=start, stop=end, step=1)
        anomaly_time = np.arange(start=index * 48, stop=(index+1)*48, step=1)

        plt.plot(time_values, full_ground_truth[start:end])
        plt.plot(anomaly_time, full_ground_truth[index * 48:(index+1)*48], color="r")
        plt.xlabel(f"Anomaly on date: {dates[(index - 1) * 48]}")

        plt.show()


def test_model(trained_model, data):
    prediction = trained_model(data)
    error = tf.losses.MeanSquaredError()(prediction, data)
    print(f"Error on test data is: {error}")


if __name__ == "__main__":
    config = Config()
    model = AE(config)

    train_data, valid_data, test_data, unshuffled_full_data, dates = load(config)

    input_shape = train_data.shape[1:]

    model.my_build(input_shape)
    model.compile(optimizer=config.optimizer, run_eagerly=True, loss="mse")

    history_dict = model.fit(x=train_data,
                             y=train_data,
                             epochs=config.num_epochs,
                             validation_data=(valid_data, valid_data),
                             # batch_size=50,
                             callbacks=[
                                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
                             ],
                             )

    plot_history(history_dict)

    test_model(model, test_data)

    plot_true_and_prediction(model, unshuffled_full_data)

    threshold = set_anomaly_threshold(model, train_data)
    find_anomalies(model, unshuffled_full_data, threshold, dates)

    print("Everything is finished")
