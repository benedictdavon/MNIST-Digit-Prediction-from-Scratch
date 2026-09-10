# Results provenance

The figures under `results/` are historical local outputs from the coursework
experiments. They are retained as artifacts, but this repository does not claim
that the reported accuracy is reproduced by the current checkout.

The reported CNN result of approximately 99.1% and the older neural-network
results were produced with particular MNIST downloads, seeds, hyperparameters,
and local environments. No full training run was performed as part of this
professionalization pass. The smoke tests cover numerical behavior of layers
and the optimizer; they are not a replacement for a training benchmark.

For a fresh CPU check, install `requirements.txt` and run:

```bash
python test_simple_cnn.py
```

The training entry points additionally require TensorFlow for dataset loading
and, for optional acceleration, a compatible CuPy/CUDA installation. The CNN
script now falls back to the NumPy backend when CuPy or CUDA is unavailable.
