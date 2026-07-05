"""Visualization helpers for MNIST predictions and convolutional activations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model


def plot_prediction_examples(
    x_test: np.ndarray,
    y_test_labels: np.ndarray,
    predicted_labels: np.ndarray,
    indices: Iterable[int],
    output_path: Path | None = None,
) -> None:
    """Plot selected prediction examples in a 3x3 grid."""
    plt.figure(figsize=(9, 9))
    for i, index in enumerate(list(indices)[:9]):
        plt.subplot(3, 3, i + 1)
        image = x_test[index].reshape(28, 28)
        plt.imshow(image, cmap="gray", interpolation="none")
        plt.title(f"Predicted {predicted_labels[index]}, Class {y_test_labels[index]}")
        plt.axis("off")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()


def get_predicted_classes(model: Model, x_test: np.ndarray) -> np.ndarray:
    """Return the highest-probability predicted class for each input image."""
    probabilities = model.predict(x_test)
    return np.argmax(probabilities, axis=1)


def visualize_layer_activations(
    model: Model,
    layer_name: str,
    image: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Visualize the feature maps produced by a named convolution/pooling layer."""
    activation_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    activations = activation_model.predict(image)
    activations = np.squeeze(activations)

    if activations.ndim != 3:
        raise ValueError(f"Layer {layer_name!r} did not produce 3D feature maps.")

    num_filters = activations.shape[-1]
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    plt.figure(figsize=(15, 12))
    for i in range(num_filters):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        ax.imshow(activations[:, :, i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
