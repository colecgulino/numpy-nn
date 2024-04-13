"""Test for a dense network."""

import unittest

import numpy as np
from numpy import testing as np_test
import torch
from torch import nn

import dense as _dense


def softmax(x):
    max_val = np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x - max_val)
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)


class TestDense(unittest.TestCase):

    def test_equal_to_pytorch(self):

        # Forward pass.
        x = torch.ones((5, 10, 10))

        dense = _dense.Dense(10, 5)
        dense2 = _dense.Dense(5, 5, name='Dense2')

        l = nn.Linear(10, 5)
        l2 = nn.Linear(5, 5)
        parameters = {
            'Dense/W': l.weight.T.detach().numpy(),
            'Dense/b': l.bias.detach().numpy(),
            'Dense2/W': l2.weight.T.detach().numpy(),
            'Dense2/b': l2.bias.detach().numpy()
        }

        dense.update_parameters(parameters)
        dense2.update_parameters(parameters)

        out_torch = l2(l(x))
        out_numpy = dense2(dense(x.numpy()))

        with self.subTest('ForwardSame'):
            np_test.assert_allclose(
                out_numpy, out_torch.detach().numpy(), atol=1e-6, rtol=1e-6
            )

        # Backward pass.
        y = torch.tensor([0., 1., 0., 0., 0.])[None, None].repeat(5, 10, 1)

        loss = nn.CrossEntropyLoss(reduction='sum')(
            out_torch.contiguous().view(-1, 5), y.contiguous().view(-1, 5)
        )
        loss.backward()
        reference_gradients = {
            'Dense/W': l.weight.grad.T,
            'Dense/b': l.bias.grad,
            'Dense2/W': l2.weight.grad.T,
            'Dense2/b': l2.bias.grad
        }

        gradients = {}
        backward_grad = softmax(out_numpy) - y.detach().numpy()
        backward_grad = dense2.backward(dense(x), {}, backward_grad, gradients)
        backward_grad = dense.backward(x, {}, backward_grad, gradients)

        for grad_name, grad_value in reference_gradients.items():
            np_test.assert_allclose(
                grad_value.detach().numpy(), gradients[grad_name],
                atol=1e-5, rtol=1e-5
            )


if __name__ == '__main__':
    unittest.main()
