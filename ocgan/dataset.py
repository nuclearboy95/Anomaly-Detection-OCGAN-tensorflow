from tensorflow import keras
import numpy as np


def get_mnist(cls=1):
    d_train, d_test = keras.datasets.mnist.load_data()
    x_train, y_train = d_train
    x_test, y_test = d_test

    mask = y_train == cls

    x_train = x_train[mask]
    x_train = np.expand_dims(x_train / 255., axis=-1).astype(np.float32)
    x_test = np.expand_dims(x_test / 255., axis=-1).astype(np.float32)

    y_test = (y_test == cls).astype(np.float32)
    return x_train, x_test, y_test  # y_test: normal -> 1 / abnormal -> 0
