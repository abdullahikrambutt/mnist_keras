"""Data loading and preprocessing helpers for MNIST."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


ArrayPair = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[ArrayPair, ArrayPair]


def load_dense_data(num_classes: int = 10) -> Dataset:
    """Load MNIST for a fully connected model.

    Images are flattened from 28x28 matrices to 784-length vectors and
    normalized to the [0, 1] range.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60_000, 784).astype("float32") / 255.0
    x_test = x_test.reshape(10_000, 784).astype("float32") / 255.0

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def load_cnn_data(num_classes: int = 10) -> Dataset:
    """Load MNIST for a convolutional neural network.

    Images are reshaped to 28x28x1 so convolutional layers can operate on the
    spatial structure of each digit image.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60_000, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(10_000, 28, 28, 1).astype("float32") / 255.0

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)
