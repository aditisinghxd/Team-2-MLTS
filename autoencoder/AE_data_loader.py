import os
import tensorflow as tf
from data.data_loader import read_taxi, read_wind
import numpy as np
import pandas as pd


def load(config):
    """
    Data loading "interface" for the autoencoder
    :param config: config with relevant information like what data set to load
    :return: batched train, valid, test,
             and complete unshuffled data set and dates
    """
    data_type = config.dataset

    if data_type == "random":
        return load_random()

    if data_type == "taxi":
        return load_taxi()

    if data_type == "wind":
        return load_wind()


def load_taxi():
    """
    Receive the data set list from the csv file.
    Doing preprocessing like normalizing and splitting the data set.
    10320 data points are split into 215 sequences with length 48, e.g. one sequence represents one day.
    :return: ()
    """
    full_data = read_taxi()
    taxi_array = np.array(full_data)
    taxi_values = np.array([float(x) for x in taxi_array[1:, 2]]).reshape([10320, 1])
    dates = [x for x in taxi_array[1:, 1]]

    taxi_values = (taxi_values - np.min(taxi_values)) / (np.max(taxi_values) - np.min(taxi_values))

    train, valid, test, unshuffled = train_valid_test_split(taxi_values, 215)

    return train, valid, test, unshuffled, dates


def load_wind():
    full_data = read_wind()
    wind_array = np.array(full_data)

    # This is not probably not optimal, but cba
    # Removing NA values in the data set and replace them by interpolation
    nan_data_array = np.array(np.where(wind_array == "NA", float("nan"), wind_array)[1:, 1:], dtype=np.float32)
    cleaned_wind_array = np.array(pd.DataFrame(nan_data_array).interpolate())
    dates = [x[0] for x in full_data[1:]]

    del full_data, wind_array

    max = np.max(cleaned_wind_array, axis=0)
    min = np.min(cleaned_wind_array, axis=0)
    normalized_wind_array = (cleaned_wind_array - min) / (max - min)

    dates = dates[:-22]
    normalized_wind_array = normalized_wind_array[:-22]
    train, valid, test, unshuffled = train_valid_test_split(normalized_wind_array, 234)

    return train, valid, test, unshuffled, dates


def train_valid_test_split(normalized_data, num_samples):
    data_batched = np.split(normalized_data, num_samples)
    unshuffled_data = data_batched.copy()
    np.random.shuffle(data_batched)

    num_train_data = int(len(data_batched) * 0.7)

    train = tf.constant(data_batched[:num_train_data])
    valid = tf.constant(data_batched[num_train_data:])

    num_valid_data = int(valid.shape[0] * 0.6666)

    test = valid[num_valid_data:]
    valid = valid[:num_valid_data]

    return train, valid, test, tf.constant(unshuffled_data)


def load_random():
    train = tf.random.uniform([1000, 256, 1])
    # train_dataset = tf.data.Dataset.from_tensor_slices(train)
    # train_dataset = train.batch(100)

    val = tf.random.uniform([300, 256, 1])
    # valid_dataset = tf.data.Dataset.from_tensor_slices(val)
    # valid_dataset = valid_dataset.batch(100)

    test = tf.random.uniform([100, 256, 1])
    return train, val, test
