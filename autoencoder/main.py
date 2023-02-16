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
    """
    Create and show plot with train and valid loss over the course of the training.
    :param history: History callback with history dict with "loss" and "val_loss" list.
    :return:
    """
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()


def plot_true_and_prediction(trained_model, true_data):
    """
    Plotting and showing Ground truth data and prediction from the trained model.
    Set to show seven consecutive time sequences in total, start is randomly chosen.
    :param trained_model: Trained model
    :param true_data: Any data set of the form [num_sequences, sequence_length, channels]
    :return:
    """
    prediction = trained_model(true_data)

    # Randomly choosing start point and computing start and finish for the whole seven sequences
    # while drawing vertical lines to separate the sequences
    weeks = true_data.shape[0]
    # 7 Sequences
    samples_for_viz = 7 * config.period_steps
    for week_start in [0,
                       config.period_steps,
                       2 * config.period_steps,
                       3 * config.period_steps,
                       4 * config.period_steps,
                       5 * config.period_steps,
                       6 * config.period_steps]:
        plt.axvline(x=week_start, color='g')
    week = np.random.randint(low=0, high=weeks, size=[1], dtype=int)[0]
    start = week * config.period_steps
    end = start + samples_for_viz

    # We are only interested in the wind speed part of the data set
    # Doesn't do anything for the taxi data set
    # And reshaping the data set to one long sequence [num_samples, channels]
    true_data = true_data[:, :, 0]
    prediction = prediction[:, :, 0]
    shape = true_data.shape
    samples = shape[0] * shape[1]
    full_ground_truth = tf.reshape(true_data, samples)
    full_prediction = tf.reshape(prediction, samples)

    # Plotting stuff
    plt.plot(full_ground_truth[start:end], label="Ground Truth")
    plt.plot(full_prediction[start:end], label="Prediction")
    plt.legend()
    plt.ylabel("Normalised data point value")
    plt.xlabel("Sample number in given week")
    plt.show()


def set_anomaly_threshold(trained_model, training_data):
    """
    Compute based on the provided data set, where the threshold for the anomaly detection is set.
    Reconstruction error split into 100 buckets that are equally likely, e.g. every bucket has 1% Chance.
    Threshold is based on the error for the highest 1%, e.g. 99% of the data have lower reconstruction error.
    :param trained_model: Trained model
    :param training_data: Preferably the data the model was trained on
    :return:
    """
    prediction = trained_model(training_data)
    loss = tf.reduce_mean(tf.losses.MeanAbsoluteError(reduction="none")(prediction, training_data), axis=1)
    return tfp.stats.quantiles(loss, 100, interpolation="linear")[-2]


def find_anomalies(trained_model, data, threshold, dates):
    """
    Find anomalies in given data set based on the provided threshold.
    :param trained_model: Trained model
    :param data: Data where to find anomalies
    :param threshold: Reconstruction threshold to decide which are anomalies
    :param dates: List of dates for the data set for displaying purposes
    :return:
    """
    # Compute prediction and compute sequence wise loss
    prediction = trained_model(data)
    loss = tf.reduce_mean(tf.losses.MeanAbsoluteError(reduction="none")(prediction, data), axis=1)

    # Plot a histogram for all losses present in the provided data set
    # Draw a vertical line where the threshold for anomaly detection is.
    plt.hist(loss, bins=40)
    plt.axvline(x=threshold, color='r', label='Anomaly threshold')
    if config.dataset == "taxi":
        plt.xlabel("Reconstruction loss in a day")
    elif config.dataset == "wind":
        plt.xlabel("Reconstruction loss in a week")
    plt.ylabel("Number of samples with loss")
    plt.show()

    # Finde sequences where the reconstruction loss is higher than the threshold
    anomalies = loss >= threshold
    anomaly_indices = np.where(anomalies)
    print(f"Number of anomalies: {np.sum(anomalies)}")
    print(f"Day[s] where anomaly occurred: {anomaly_indices}")

    # Some data fiddling not important for taxi data set
    # Important for wind data set, we are interested in wind speed
    data = data[:, :, 0]
    prediction = prediction[:, :, 0]
    shape = data.shape
    num_samples = shape[0] * shape[1]
    full_ground_truth = tf.reshape(data, num_samples)
    prediction = tf.reshape(prediction, num_samples)

    # Create a plot for every anomalous sequence.
    # Not only the anomaly os plotted, but also sequences before and after the anomaly.
    for index in anomaly_indices[0]:
        start, end = get_start_end(num_samples, index)

        time_values = np.arange(start=start, stop=end, step=1)
        anomaly_time = np.arange(start=index * shape[1], stop=(index + 1) * shape[1], step=1)

        plt.plot(time_values, full_ground_truth[start:end],
                 color="b",
                 label="Ground Truth")
        plt.plot(anomaly_time, full_ground_truth[index * shape[1]:(index + 1) * shape[1]],
                 color="r",
                 label="Detected Anomaly")
        plt.plot(anomaly_time, prediction[index * shape[1]:(index + 1) * shape[1]], label="Reconstruction")
        plt.xlabel(f"Anomaly on date: {dates[index * shape[1]]}")
        plt.legend()

        plt.show()


def get_start_end(num_samples, index):
    """
    Compute where to start and where to end a time sequence when provided with the number of a sequence.
    The time sequence is supposed to start before and end after the "index-sequence".
    :param num_samples: number of total samples in the sequence
    :param index: index of the time sequence we are interested in
    :return: (start, end) of the time sequence
    """
    if (index + 3) * config.period_steps > num_samples:
        start = (index - 7) * config.period_steps
        end = index * config.period_steps
    elif index <= 3:
        start = 0
        end = 7 * config.period_steps
    else:
        start = (index - 3) * config.period_steps
        end = (index + 4) * config.period_steps
    return start, end


def test_model(trained_model, data):
    """
    Computing total loss on given data
    :param trained_model: Trained model
    :param data: Data to compute error on.
                [num_sequences, sequence_length, channels]
    :return:
    """
    prediction = trained_model(data)
    error = my_loss(data, prediction)
    print(f"Error on test data is: {error}")


def my_loss(true, pred):
    """
    Standard MeanSquareError for the Taxi data set.
    Custom loss function needed for the wind data set.
    Weighting the reconstruction of the wind speed ten times higher then the rest.
    :param true: Ground Truth
    :param pred: Prediction from the model
    :return: (Weighted) mean square error
    """
    if config.dataset == "taxi":
        mse = tf.keras.losses.MeanSquaredError()
        return mse(true, pred)
    elif config.dataset == "wind":
        error = (true - pred)[:, :, 0] * 10
        return tf.reduce_mean(tf.square(error))


if __name__ == "__main__":
    """
    1. Loading data
    2. Creating and compiling model
    3. Training
    4. Evaluating
    """
    # 1.
    train_data, valid_data, test_data, unshuffled_full_data, dates = load(config)

    input_shape = train_data.shape[1:]

    # 2.
    model = AE(config)
    model.my_build(input_shape)
    model.compile(optimizer=config.optimizer, run_eagerly=True, loss=my_loss)

    # 3.
    history_dict = model.fit(x=train_data,
                             y=train_data,
                             epochs=config.num_epochs,
                             validation_data=(valid_data, valid_data),
                             # batch_size=50,
                             callbacks=[
                                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
                             ],
                             )

    # 4.
    plot_history(history_dict)

    test_model(model, test_data)

    plot_true_and_prediction(model, unshuffled_full_data)

    threshold = set_anomaly_threshold(model, train_data)
    find_anomalies(model, unshuffled_full_data, threshold, dates)

    print("Everything is finished")
