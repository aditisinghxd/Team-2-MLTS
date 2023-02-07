import os
import tensorflow as tf
from data.data_loader import read_taxi, read_wind


def load(config):
    data_type = config.dataset
    if data_type == "random":
        return load_random()
    if data_type == "taxi":
        full_data = read_taxi()
    if data_type == "wind":
        full_data = read_wind()

    return


def load_random():
    train = tf.random.uniform([1000, 256, 1])
    # train_dataset = tf.data.Dataset.from_tensor_slices(train)
    # train_dataset = train_dataset.batch(100)

    val = tf.random.uniform([300, 256, 1])
    # valid_dataset = tf.data.Dataset.from_tensor_slices(val)
    # valid_dataset = valid_dataset.batch(100)
    return train, val
