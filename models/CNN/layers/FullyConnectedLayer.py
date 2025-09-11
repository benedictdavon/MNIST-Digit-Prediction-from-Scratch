import numpy as np
from .Layer import Layer
from ..helpers.Backend import backend


class FullyConnectedLayer(Layer):
    def __init__(self, in_features, out_features):
        # initialize weights/bias
        # weights: (out_features, in_features)
        # bias: (out_features, 1)
        self.in_features = in_features
        self.out_features = out_features

        # He initialization
        # use float32 for efficiency
        dtype = np.float32

        # Initialize on CPU, then move to backend
        weights_cpu = (
            np.random.randn(out_features, in_features).astype(dtype)
            * np.sqrt(2.0 / in_features)
        ).astype(dtype)
        bias_cpu = np.zeros((out_features, 1), dtype=dtype)

        self.weights = backend.ensure_array(weights_cpu)
        self.bias = backend.ensure_array(bias_cpu)

        self.dW = backend.zeros_like(self.weights)
        self.db = backend.zeros_like(self.bias)

    def forward(self, x, training=False):
        # x shape: (batch, in_features)
        # return: (batch, out_features)
        self.x = backend.ensure_array(x)  # cache for backward
        return backend.matmul(self.x, backend.transpose(self.weights)) + backend.transpose(self.bias)

    def backward(self, grad_out):
        grad_out = backend.ensure_array(grad_out)
        self.dW[...] = backend.matmul(backend.transpose(grad_out), self.x)  # (out, in)
        self.db[...] = backend.transpose(backend.sum(grad_out, axis=0, keepdims=True))
        return backend.matmul(grad_out, self.weights)  # (B, in)

    def params(self):
        return [self.weights, self.bias]

    def grads(self):
        return [self.dW, self.db]
