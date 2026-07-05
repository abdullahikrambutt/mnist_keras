"""Model architectures for MNIST classification."""

from __future__ import annotations

from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential


def build_dense_model(input_shape: tuple[int, ...] = (784,), num_classes: int = 10) -> Sequential:
    """Build the fully connected MNIST model from the recreated notebook."""
    model = Sequential(
        [
            Dense(512, input_shape=input_shape),
            Activation("relu"),
            Dropout(0.2),
            Dense(512),
            Activation("relu"),
            Dropout(0.2),
            Dense(num_classes),
            Activation("softmax"),
        ],
        name="mnist_dense_classifier",
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_cnn_model(input_shape: tuple[int, ...] = (28, 28, 1), num_classes: int = 10) -> Sequential:
    """Build the convolutional MNIST model from the recreated notebook."""
    model = Sequential(name="mnist_cnn_classifier")

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, name="conv_1"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu", name="relu_1"))

    model.add(Conv2D(32, (3, 3), name="conv_2"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu", name="relu_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_1"))

    model.add(Conv2D(64, (3, 3), name="conv_3"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu", name="relu_3"))

    model.add(Conv2D(64, (3, 3), name="conv_4"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu", name="relu_4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_2"))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
