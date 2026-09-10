from __future__ import annotations

import unittest

import numpy as np

from models.CNN.helpers.Backend import backend
from models.CNN.layers.Conv2D import Conv2D
from models.CNN.layers.FullyConnectedLayer import FullyConnectedLayer
from models.CNN.layers.MaxPool2D import MaxPool2D
from models.CNN.optimizer.AdamWOptimizer import AdamWOptimizer


class CoreNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These are deterministic CPU unit tests. GPU behavior is exercised by
        # the same backend abstraction when the training script is run with a
        # working CuPy/CUDA installation.
        backend.use_gpu = False
        backend.xp = np

    def test_conv_forward_backward_shape_and_finite_values(self):
        layer = Conv2D(1, 2, kernel_size=3)
        x = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5) / 25.0
        output = layer.forward(x)
        gradient = layer.backward(np.ones_like(output))

        self.assertEqual(output.shape, (1, 2, 3, 3))
        self.assertEqual(gradient.shape, x.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.isfinite(gradient).all())

    def test_max_pool_backward_routes_gradient_to_maximum(self):
        layer = MaxPool2D(kernel_size=2, stride=2)
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        output = layer.forward(x)
        gradient = layer.backward(np.ones_like(output))

        np.testing.assert_array_equal(output, [[[[4.0]]]])
        np.testing.assert_array_equal(gradient, [[[[0.0, 0.0], [0.0, 1.0]]]])

    def test_fully_connected_gradient_matches_finite_difference(self):
        layer = FullyConnectedLayer(3, 2)
        x = np.array([[0.2, -0.4, 0.7]], dtype=np.float32)
        upstream = np.array([[0.3, -0.8]], dtype=np.float32)
        layer.forward(x)
        layer.backward(upstream)
        analytic = float(layer.dW[0, 1])

        original = float(layer.weights[0, 1])
        epsilon = 1e-3
        layer.weights[0, 1] = original + epsilon
        plus = float(np.sum(layer.forward(x) * upstream))
        layer.weights[0, 1] = original - epsilon
        minus = float(np.sum(layer.forward(x) * upstream))
        layer.weights[0, 1] = original

        numerical = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(analytic, numerical, places=3)

    def test_adamw_updates_parameter_and_keeps_state_shape(self):
        parameter = np.array([1.0, -2.0], dtype=np.float32)
        gradient = np.array([0.5, -0.25], dtype=np.float32)
        optimizer = AdamWOptimizer([[parameter, gradient]], lr=0.1, weight_decay=0.01)
        before = parameter.copy()

        optimizer.step()

        self.assertFalse(np.array_equal(parameter, before))
        self.assertEqual(optimizer._m[id(parameter)].shape, parameter.shape)
        self.assertEqual(optimizer._v[id(parameter)].shape, parameter.shape)
        self.assertTrue(np.isfinite(parameter).all())


if __name__ == "__main__":
    unittest.main()
