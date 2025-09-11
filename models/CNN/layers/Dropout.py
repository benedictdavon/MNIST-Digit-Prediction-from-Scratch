import numpy as np
from .Layer import Layer
from ..helpers.Backend import backend

class Dropout(Layer):
    def __init__(self, p=0.1):
        self.p = float(p)
        self.mask = None

    def forward(self, x, training=False):
        x = backend.ensure_array(x)
        if training and self.p > 0.0:
            keep = 1.0 - self.p
            # Generate random mask using the same backend as input
            if backend.use_gpu:
                # Use CuPy's random number generator
                random_vals = backend.random.rand(*x.shape)
            else:
                # Use NumPy's random number generator
                random_vals = backend.random.rand(*x.shape)
            
            # Ensure mask is on the same backend as input
            mask_bool = random_vals < keep
            self.mask = backend.ensure_array(mask_bool.astype(x.dtype)) / keep
            return x * self.mask
        self.mask = None
        return x

    def backward(self, grad_out):
        grad_out = backend.ensure_array(grad_out)
        return grad_out if self.mask is None else grad_out * self.mask

    def params(self): return []
    def grads(self):  return []
