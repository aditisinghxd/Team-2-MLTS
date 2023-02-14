import os
from config import Config
from AE import AE
from AE_data_loader import load
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
import numpy as np

config = Config()


def plot_history(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()


def plot_true_and_prediction(trained_model, true_data):
    prediction = trained_model(true_data)

    if config.dataset == "wind":
        weeks = true_data.shape[0]
        # 7 Weeks
        samples_for_viz = 7*7
        for week_start in [7, 2 * 7, 3 * 7, 4 * 7, 5 * 7, 6 * 7]:
            plt.axvline(x=week_start, color='r')
        week = np.random.randint(low=0, high=weeks, size=[1], dtype=int)[0]
        start = week * 7
        end = start + samples_for_viz

    elif config.dataset == "taxi":
        days = true_data.shape[0]
        # 7 Days
        samples_for_viz = 7*48
        for day_start in [48, 2 * 48, 3 * 48, 4 * 48, 5 * 48, 6 * 48]:
            plt.axvline(x=day_start, color='r')
        day = np.random.randint(low=0, high=days, size=1, dtype=int)[0]
        start = day * 48
        end = start + samples_for_viz
    else:
        print("Neither taxi nor wind dataset")
        return

    true_data = true_data[:, :, 0]
    prediction = prediction[:, :, 0]

    shape = true_data.shape
    samples = shape[0] * shape[1]
    full_ground_truth = tf.reshape(true_data, samples)
    full_prediction = tf.reshape(prediction, samples)

    plt.plot(full_ground_truth[start:end], label="Ground Truth")
    plt.plot(full_prediction[start:end], label="Prediction")

    plt.legend()
    if config.dataset == "taxi":
        plt.title("Plot for one whole week")
    elif config.dataset == "wind":
        plt.title("Plot for seven weeks")
    plt.show()


def set_anomaly_threshold(trained_model, training_data):
    prediction = trained_model(training_data)
    loss = tf.reduce_mean(tf.losses.MeanAbsoluteError(reduction="none")(prediction, training_data), axis=1)
    return tfp.stats.quantiles(loss, 100, interpolation="linear")[-2]


def find_anomalies(trained_model, data, threshold, dates):
    prediction = trained_model(data)
    loss = tf.reduce_mean(tf.losses.MeanAbsoluteError(reduction="none")(prediction, data), axis=1)

    plt.hist(loss, bins=40)
    plt.axvline(x=threshold, color='r', label='Anomaly threshold')
    if config.dataset == "taxi":
        plt.xlabel("Reconstruction loss in a day")
    elif config.dataset == "wind":
        plt.xlabel("Reconstruction loss in a week")
    plt.ylabel("Number of samples with loss")
    plt.show()

    anomalies = loss >= threshold
    anomaly_indices = np.where(anomalies)
    print(f"Number of anomalies: {np.sum(anomalies)}")
    print(f"Day[s] where anomaly occurred: {anomaly_indices}")

    data = data[:, :, 0]
    shape = data.shape
    num_samples = shape[0] * shape[1]
    full_ground_truth = tf.reshape(data, num_samples)
    for index in anomaly_indices[0]:
        start, end = get_start_end(num_samples, index)

        time_values = np.arange(start=start, stop=end, step=1)
        anomaly_time = np.arange(start=index * shape[1], stop=(index+1)*shape[1], step=1)

        plt.plot(time_values, full_ground_truth[start:end])
        plt.plot(anomaly_time, full_ground_truth[index * shape[1]:(index+1)*shape[1]], color="r")
        plt.xlabel(f"Anomaly on date: {dates[index * shape[1]]}")

        plt.show()


def get_start_end(num_samples, index):
    if config.dataset == "taxi":
        if (index + 3) * 48 > num_samples:
            start = (index - 7) * 48
            end = index * 48
        elif index <= 3:
            start = 0
            end = 7 * 48
        else:
            start = (index - 3) * 48
            end = (index + 4) * 48
        return start, end

    elif config.dataset == "wind":
        if (index + 3) * 7 > num_samples:
            start = (index - 7) * 7
            end = index * 7
        elif index <= 3:
            start = 0
            end = 7 * 7
        else:
            start = (index - 3) * 7
            end = (index + 4) * 7
        return start, end


def test_model(trained_model, data):
    prediction = trained_model(data)
    error = tf.losses.MeanSquaredError()(prediction, data)
    print(f"Error on test data is: {error}")


if __name__ == "__main__":
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
