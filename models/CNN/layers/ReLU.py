import numpy as np
from .Layer import Layer
from ..helpers.Backend import backend

class ReLU(Layer):
    def forward(self, x, training=False):
        x = backend.ensure_array(x)
        self.mask = (x > 0).astype(np.float32)
        self.mask = backend.ensure_array(self.mask)
        return backend.maximum(0, x)

    def backward(self, grad_out):
        grad_out = backend.ensure_array(grad_out)
        return grad_out * self.mask
