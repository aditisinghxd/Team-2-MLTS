import os
import tensorflow as tf
from data.data_loader import read_taxi, read_wind
import numpy as np


def load(config):
    data_type = config.dataset

    if data_type == "random":
        return load_random()

    if data_type == "taxi":
        return load_taxi()

    if data_type == "wind":
        return load_wind()


def load_taxi():
    full_data = read_taxi()
    taxi_array = np.array(full_data)
    taxi_values = np.array([float(x) for x in taxi_array[1:, 2]])[:-2].reshape([10318, 1])

    taxi_values /= np.max(taxi_values)

    taxi_batched = np.split(taxi_values, 1474)
    np.random.shuffle(taxi_batched)

    num_train_data = int(len(taxi_batched) * 0.7)

    taxi_train = tf.constant(taxi_batched[:num_train_data])
    taxi_valid = tf.constant(taxi_batched[num_train_data:])

    num_valid_data = int(taxi_valid.shape[0] * 0.6666)

    taxi_test = taxi_valid[num_valid_data:]
    taxi_valid = taxi_valid[:num_valid_data]
    return taxi_train, taxi_valid, taxi_test, tf.constant(taxi_batched)


def load_wind():
    full_data = read_wind()
    wind_array = np.array(full_data)
    wind_wind = [float(x) for x in wind_array]
    return


def load_random():
    train = tf.random.uniform([1000, 256, 1])
    # train_dataset = tf.data.Dataset.from_tensor_slices(train)
    # train_dataset = train.batch(100)

    val = tf.random.uniform([300, 256, 1])
    # valid_dataset = tf.data.Dataset.from_tensor_slices(val)
    # valid_dataset = valid_dataset.batch(100)

    test = tf.random.uniform([100, 256, 1])
    return train, val, test
