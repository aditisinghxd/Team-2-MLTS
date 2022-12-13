import os
import tensorflow as tf


def load(config):
    data_type = config.dataset
    if data_type == "zeros":
        return load_all_zeros()


def load_all_zeros():
    dataset = tf.data.Dataset.from_tensor_slices(tf.zeros([1000, 32, 32, 3]))
    dataset = dataset.batch(100)
    return dataset
