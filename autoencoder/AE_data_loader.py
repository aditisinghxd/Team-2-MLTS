import os
import tensorflow as tf
from data.data_loader import read_taxi, read_wind
import numpy as np


def load(config):
    data_type = config.dataset
    if data_type == "random":
        return load_random()

    if data_type == "taxi":
        full_data = read_taxi()
        taxi_array = np.array(full_data)
        taxi_values = np.array([int(x) for x in taxi_array[1:, 2]])[:-16].reshape([10304, 1])

        taxi_batched = np.split(taxi_values, 368)
        np.random.shuffle(taxi_batched)

        taxi_train = tf.constant(taxi_batched[:300])
        taxi_valid = tf.constant(taxi_batched[300:])
        return taxi_train, taxi_valid

    if data_type == "wind":
        full_data = read_wind()
        wind_array = np.array(full_data)
        wind_wind = [float(x) for x in wind_array]




def load_random():
    train = tf.random.uniform([1000, 256, 1])
    # train_dataset = tf.data.Dataset.from_tensor_slices(train)
    # train_dataset = train.batch(100)

    val = tf.random.uniform([300, 256, 1])
    # valid_dataset = tf.data.Dataset.from_tensor_slices(val)
    # valid_dataset = valid_dataset.batch(100)
    return train, val
