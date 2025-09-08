# CNN Plan

## Architecture (baseline)
- Input: (N, 1, 28, 28) MNIST grayscale images
- Conv layer: 8 filters, 3×3 kernel, stride 1, no padding → output (N, 8, 26, 26)
- ReLU activation
- Max Pool: 2×2, stride 2 → output (N, 8, 13, 13)
- Flatten → (N, 1352)
- Fully connected layer: 128 hidden units, ReLU
- Output: 10 units (digits 0–9), softmax
- Loss: cross-entropy
- Optimizer: Adam
- Regularization: optional L2 and dropout

## Targets
- Test accuracy ≥ 98.5%
- Converges within 20–30 epochs
- Log training/validation loss & accuracy
- Save confusion matrix and metrics

## Implementation Steps
1. Convolution forward
2. Pooling forward
3. Flatten + dense + softmax forward
4. Backward (dense → pool → conv)
5. Adam update
6. Training loop with early stopping
7. Documentation + plots + README update
