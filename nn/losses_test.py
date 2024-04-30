"""Test for losses in Numpy NN library."""

import unittest

from numpy import testing as np_test
import torch
from torch import nn

from nn import activations
from nn import loss_base
from nn import losses
from nn import dense as _dense


def check_gradient(
        loss_fn: loss_base.Loss,
        torch_loss_fn: nn.Module
) -> None:
    """Checks gradients for a given loss function."""
    x = torch.ones((5, 10, 10))

    dense = _dense.Dense(10, 5)

    l = nn.Linear(10, 5)
    parameters = {
        'Dense/W': l.weight.T.detach().numpy(),
        'Dense/b': l.bias.detach().numpy(),
    }

    dense.update_parameters(parameters)

    out_torch = nn.ReLU()(l(x))
    activation = activations.ReLU()
    out_numpy = activation(dense(x.numpy()))

    # Backward pass.
    y = torch.tensor([0., 1., 0., 0., 0.])[None, None].repeat(5, 10, 1)

    torch_loss = torch_loss_fn(
        out_torch.contiguous().view(-1, 5), y.contiguous().view(-1, 5)
    )
    torch_loss.backward()
    reference_gradients = {
        'Dense/W': l.weight.grad.T,
        'Dense/b': l.bias.grad,
    }

    gradients = {}
    backward_grad = loss_fn.backward(out_numpy, y.detach().numpy())
    backward_grad = activation.backward(dense(x.numpy()), {}, backward_grad, gradients)
    backward_grad = dense.backward(x.numpy(), {}, backward_grad, gradients)

    for grad_name, grad_value in reference_gradients.items():
        np_test.assert_allclose(
            grad_value.detach().numpy(), gradients[grad_name],
            atol=1e-5, rtol=1e-5
        )


class TestMSE(unittest.TestCase):
    """Test mean squared error loss."""

    def test_forward(self):
        """Test forward pass."""
        x = torch.ones((5, 5, 10))
        y = torch.zeros((5, 5, 10))

        with self.subTest('Mean'):
            loss_fn = losses.MSE(reduction=loss_base.Reduction.MEAN, dim=-1)
            np_test.assert_allclose(
                loss_fn(x.detach().numpy(), y.detach().numpy()),
                nn.MSELoss()(x, y)
            )

        with self.subTest('Mean'):
            loss_fn = losses.MSE(reduction=loss_base.Reduction.SUM, dim=-1)
            np_test.assert_allclose(
                loss_fn(x.detach().numpy(), y.detach().numpy()),
                nn.MSELoss(reduction='sum')(x, y)
            )

    def test_backward(self):
        """Test backward pass."""
        with self.subTest('Mean'):
            check_gradient(losses.MSE(dim=-1), nn.MSELoss())
        with self.subTest('Sum'):
            check_gradient(
                losses.MSE(reduction=loss_base.Reduction.SUM, dim=-1),
                nn.MSELoss(reduction='sum')
            )


class TestCrossEntropy(unittest.TestCase):
    """Test cross entropy loss function."""

    def test_forward(self):
        """Test forward pass."""
        x = torch.ones((6, 10, 5))
        y = torch.tensor([[[0., 1., 0., 0., 0.]]]).repeat([6, 10, 1])
        with self.subTest('Mean'):
            loss_fn = losses.CrossEntropy(reduction=loss_base.Reduction.MEAN, dim=-1)
            np_test.assert_allclose(
                loss_fn(x.detach().numpy(), y.detach().numpy()),
                nn.CrossEntropyLoss()(x.reshape(-1, 5), y.reshape(-1, 5)),
                atol=1e-6, rtol=1e-6
            )

        with self.subTest('Sum'):
            loss_fn = losses.CrossEntropy(reduction=loss_base.Reduction.SUM, dim=-1)
            np_test.assert_allclose(
                loss_fn(x.detach().numpy(), y.detach().numpy()),
                nn.CrossEntropyLoss(reduction='sum')(x.reshape(-1, 5), y.reshape(-1, 5)),
                atol=1e-6, rtol=1e-6
            )

    def test_backward(self):
        """Test backward pass."""
        with self.subTest('Mean'):
            check_gradient(losses.CrossEntropy(dim=-1), nn.CrossEntropyLoss())
        with self.subTest('Sum'):
            check_gradient(
                losses.CrossEntropy(reduction=loss_base.Reduction.SUM, dim=-1),
                nn.CrossEntropyLoss(reduction='sum')
            )


if __name__ == '__main__':
    unittest.main()
