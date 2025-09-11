import numpy as np
from .Layer import Layer
from ..helpers.Backend import backend


class Flatten(Layer):
    def forward(self, x, training=False):
        # x shape: (batch, channels, H, W)
        # return: (batch, channels*H*W)
        x = backend.ensure_array(x)
        self.in_shape = x.shape
        return backend.reshape(x, (x.shape[0], -1))

    def backward(self, grad_out):
        grad_out = backend.ensure_array(grad_out)
        return backend.reshape(grad_out, self.in_shape)

