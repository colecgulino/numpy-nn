"""Testing general optimizers."""

import unittest

import numpy as np

import activations
import dense
import losses
import optimizers


def get_network():
    """Gets simple network for optimization."""
    layers = [
        dense.Dense(10, 10),
        activations.Sigmoid(),
        dense.Dense(10, 10, name='Dense2')
    ]
    all_parameters = {}
    for layer in layers:
        all_parameters.update(layer.parameters())
    return layers, all_parameters


class TestOptimizers(unittest.TestCase):
    """Tests optimters."""

    def test_sgd(self):
        """Tests sgd algorithm gets sufficient loss."""
        layers, all_parameters = get_network()
        loss_fn = losses.MSE(reduction='mean', dim=-1)
        optimizer2 = optimizers.SGD(1e-3, all_parameters, loss_fn)

        x = np.zeros((64, 10))
        y = np.ones((64, 10))
        for _ in range(10_000):
            output = optimizer2.step(x, y, layers)
        self.assertLessEqual(output.loss, 1e-5)

    def test_adam(self):
        """Tests adam algorithm gets sufficient loss."""
        layers, all_parameters = get_network()
        loss_fn = losses.MSE(reduction='mean', dim=-1)
        optimizer2 = optimizers.Adam(1e-3, all_parameters, loss_fn)

        x = np.zeros((64, 10))
        y = np.ones((64, 10))
        for _ in range(10_000):
            output = optimizer2.step(x, y, layers)
        self.assertLessEqual(output.loss, 1e-5)

    def test_momentum(self):
        """Tests sgd momentum gets sufficient loss"""
        layers, all_parameters = get_network()
        loss_fn = losses.MSE(reduction='mean', dim=-1)
        optimizer2 = optimizers.Momentum(1e-3, all_parameters, loss_fn)

        x = np.zeros((64, 10))
        y = np.ones((64, 10))
        for _ in range(10_000):
            output = optimizer2.step(x, y, layers)
        self.assertLessEqual(output.loss, 1e-5)


if __name__ == '__main__':
    unittest.main()
