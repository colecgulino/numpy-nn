"""Test for regularization layers."""

import unittest

import numpy as np
from numpy import testing as np_test

from nn import regularization


class TestDropout(unittest.TestCase):
    """Test dropout layer."""

    def test_forward(self):
        """Test forward pass."""
        x = np.random.random((5, 10, 100))
        do = regularization.Dropout(p=1.)
        np_test.assert_allclose(do(x), np.zeros_like(x))

    def test_backward(self):
        """Test backward pass."""
        x = np.ones((5, 10, 10))
        do = regularization.Dropout(p=0.5)
        _, cache = do.forward(x)

        backward_grad = do.backward(x, cache, x, {})
        np_test.assert_allclose(backward_grad, backward_grad * cache['mask'])


if __name__ == '__main__':
    unittest.main()
 