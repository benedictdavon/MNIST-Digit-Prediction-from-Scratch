# MNIST models from scratch

NumPy/CuPy implementations of a fully connected network and a LeNet-style CNN
for MNIST. The repository is an educational implementation project: it exposes
convolution, pooling, backpropagation, optimizers, and a small backend
abstraction instead of hiding them behind a deep-learning framework.

## What is here

- `models/NN/`: fully connected baseline and mini-batch network.
- `models/CNN/`: convolution, max-pooling, fully connected layers, loss,
  schedulers, early stopping, and SGD/AdamW optimizers.
- `MNIST_digit_predictions_CNN.py`: TensorFlow MNIST loader plus custom CNN
  training entry point.
- `test_simple_cnn.py`: CPU numerical smoke tests for layer shapes, pooling
  gradients, finite-difference gradients, and AdamW updates.

The CNN training script prefers CuPy when a working CUDA runtime is available
and otherwise uses NumPy. CuPy is optional; CPU execution does not require it.

## Setup and checks

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python test_simple_cnn.py
```

For the TensorFlow-backed MNIST entry points:

```bash
pip install -r requirements-full.txt
python MNIST_digit_predictions_CNN.py
```

Install `requirements-gpu.txt` only when the local CUDA version matches the
chosen CuPy package. The training command downloads MNIST through Keras and can
take substantially longer than the unit tests.

## Results and limits

The `results/` images are historical coursework outputs. The README previously
presented the approximately 99.1% CNN result and a claimed 10x framework speed
difference as if they were current benchmarks; those claims are now explicitly
qualified. They were not rerun here. See
[`docs/results-provenance.md`](docs/results-provenance.md).

This code is intended to make the mechanics of a small neural network
inspectable. It is not a production training framework, and comparisons with
TensorFlow/PyTorch require a controlled rerun with the same data, hardware,
seeds, and stopping policy.
