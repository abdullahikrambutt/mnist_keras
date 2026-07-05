"""Train the fully connected MNIST classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_dense_data
from .models import build_dense_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dense neural network on MNIST.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to save model and metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_dense_data()
    print("Training matrix shape:", x_train.shape)
    print("Testing matrix shape:", x_test.shape)

    model = build_dense_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }

    model_path = args.output_dir / "mnist_dense_model.keras"
    metrics_path = args.output_dir / "mnist_dense_metrics.json"
    model.save(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
