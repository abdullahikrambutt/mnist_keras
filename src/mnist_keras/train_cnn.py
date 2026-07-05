"""Train the convolutional MNIST classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .data import load_cnn_data
from .models import build_cnn_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to save model and metrics.")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable image augmentation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_cnn_data()
    print("Training matrix shape:", x_train.shape)
    print("Testing matrix shape:", x_test.shape)

    model = build_cnn_model()
    model.summary()

    if args.no_augmentation:
        history = model.fit(
            x_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_data=(x_test, y_test),
        )
    else:
        train_datagen = ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.08,
            shear_range=0.3,
            height_shift_range=0.08,
            zoom_range=0.08,
        )
        test_datagen = ImageDataGenerator()
        train_generator = train_datagen.flow(x_train, y_train, batch_size=args.batch_size)
        test_generator = test_datagen.flow(x_test, y_test, batch_size=args.batch_size)

        history = model.fit(
            train_generator,
            steps_per_epoch=len(x_train) // args.batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_data=test_generator,
            validation_steps=len(x_test) // args.batch_size,
        )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }

    model_path = args.output_dir / "mnist_cnn_model.keras"
    metrics_path = args.output_dir / "mnist_cnn_metrics.json"
    model.save(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
