"""Test for activations in Numpy NN library."""

import unittest

import numpy as np
from numpy import testing as np_test
import torch
from torch import nn

import activations
import dense as _dense
import layer


def softmax(x):
    max_val = np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x - max_val)
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)


def check_gradient(
        activation: layer.Layer,
        torch_activation: nn.Module
) -> None:
        x = torch.ones((5, 10, 10))

        dense = _dense.Dense(10, 5)

        l = nn.Linear(10, 5)
        parameters = {
            'Dense/W': l.weight.T.detach().numpy(),
            'Dense/b': l.bias.detach().numpy(),
        }

        dense.update_parameters(parameters)

        out_torch = torch_activation(l(x))
        out_numpy = activation(dense(x.numpy()))

        # Backward pass.
        y = torch.tensor([0., 1., 0., 0., 0.])[None, None].repeat(5, 10, 1)

        loss = nn.CrossEntropyLoss(reduction='sum')(
            out_torch.contiguous().view(-1, 5), y.contiguous().view(-1, 5)
        )
        loss.backward()
        reference_gradients = {
            'Dense/W': l.weight.grad.T,
            'Dense/b': l.bias.grad,
        }

        gradients = {}
        backward_grad = softmax(out_numpy) - y.detach().numpy()
        backward_grad = activation.backward(dense(x.numpy()), {}, backward_grad, gradients)
        backward_grad = dense.backward(x.numpy(), {}, backward_grad, gradients)

        for grad_name, grad_value in reference_gradients.items():
            np_test.assert_allclose(
                grad_value.detach().numpy(), gradients[grad_name],
                atol=1e-5, rtol=1e-5
            )


class TestSigmoid(unittest.TestCase):

    def test_forward(self):
        x = torch.ones((5, 5, 10))
        out = activations.Sigmoid()(x.numpy())
        out_torch = nn.Sigmoid()(x)
        np_test.assert_allclose(out, out_torch.numpy())

    def test_backward(self):
         check_gradient(activations.Sigmoid(), nn.Sigmoid())


class TestReLU(unittest.TestCase):

    def test_forward(self):
        x = torch.ones((5, 5, 10))
        out = activations.ReLU()(x.numpy())
        out_torch = nn.ReLU()(x)
        np_test.assert_allclose(out, out_torch.numpy())

    def test_backward(self):
         check_gradient(activations.ReLU(), nn.ReLU())


class TestTanh(unittest.TestCase):

    def test_forward(self):
        x = torch.ones((5, 5, 10))
        out = activations.Tanh()(x.numpy())
        out_torch = nn.Tanh()(x)
        np_test.assert_allclose(out, out_torch.numpy())

    def test_backward(self):
         check_gradient(activations.Tanh(), nn.Tanh())


class TestSoftmax(unittest.TestCase):

    def test_forward(self):
        x = torch.ones((5, 5, 10))
        out = activations.Softmax(dim=-1)(x.numpy())
        out_torch = nn.Softmax(dim=-1)(x)
        np_test.assert_allclose(out, out_torch.numpy())


if __name__ == '__main__':
    unittest.main()
