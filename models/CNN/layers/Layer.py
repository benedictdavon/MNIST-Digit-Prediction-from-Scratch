import numpy as np


class Layer:
    # Subclasses override as needed
    def forward(self, x, training=False):
        raise NotImplementedError

    def backward(self, grad_out):
        # Return grad wrt input
        raise NotImplementedError

    def params(self):
        # Return list of parameter ndarrays (e.g., [W, b])
        return []

    def grads(self):
        # Return list of gradient ndarrays matching params()
        return []