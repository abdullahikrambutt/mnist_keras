# MNIST Handwritten Digit Classification with Keras/TensorFlow

A clean, GitHub-ready recreation of an MNIST deep learning notebook using modern `tensorflow.keras` APIs.

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)

This project runs on **Python 3.10 or newer**.

This project trains two models on the MNIST handwritten digit dataset:

1. **Fully Connected Neural Network** — flattens each 28×28 image into a 784-value vector.
2. **Convolutional Neural Network** — keeps the 28×28 image shape and learns spatial image features through convolution and pooling.

The goal is to classify handwritten digits from 0 to 9 using the 60,000-image MNIST training set and the 10,000-image test set.

## Project structure

```text
mnist-keras-github-repo/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── docs/
│   └── github_setup.md
├── notebooks/
│   └── mnist_in_keras_recreated.ipynb
├── outputs/
│   └── .gitkeep
└── src/
    └── mnist_keras/
        ├── __init__.py
        ├── data.py
        ├── models.py
        ├── train_dense.py
        ├── train_cnn.py
        └── visualize.py
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate     # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install (`pip install -e .`) lets you run the package with `python -m mnist_keras...` from the project folder.

## Train the fully connected model

```bash
python -m mnist_keras.train_dense --epochs 5 --batch-size 128
```

The script will:

- download/load MNIST through Keras,
- normalize pixel values to `[0, 1]`,
- one-hot encode labels,
- train a dense neural network,
- evaluate on the test set,
- save the trained model and metrics in `outputs/`.

## Train the convolutional model

```bash
python -m mnist_keras.train_cnn --epochs 5 --batch-size 128
```

The CNN version uses convolution, batch normalization, max pooling, dropout, and light data augmentation.

## Run the notebook

```bash
jupyter notebook notebooks/mnist_in_keras_recreated.ipynb
```

## Expected results

With 5 epochs, the dense model should typically pass 95% test accuracy, and the CNN should usually perform higher. Exact results depend on hardware, TensorFlow version, random initialization, and training settings.

## Notes

- This repo uses modern `tensorflow.keras` imports instead of older standalone `keras` import paths.
- Deprecated notebook methods such as `predict_classes` and `fit_generator` were replaced with current equivalents.
- The original learning notebook content credited Daniel Moser, Xavier Snelgrove, and Yash Katariya. This repo is a cleaned and modernized recreation for GitHub use.
