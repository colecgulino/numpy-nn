"""Test for convolution layers."""

import unittest

import numpy as np
from numpy import testing as np_test
import torch
from torch import nn

import convolution
import losses
import optimizers


class Conv1DTest(unittest.TestCase):
    """Test 1D convolution."""

    def test_forward(self):
        """Test forward pass."""
        b, t = 2, 6
        x = np.random.random(size=(b, t, 10))
        conv1d_1 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        conv1d_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)

        myconv1d_1 = convolution.Conv1D(10, 20)
        myconv1d_2 = convolution.Conv1D(20, 5, name='Conv1D1')
        myconv1d_1.update_parameters({
            'Conv1D/W': conv1d_1.weight.permute([2, 1, 0]).detach().numpy(),
            'Conv1D/b': conv1d_1.bias.detach().numpy()
        })
        myconv1d_2.update_parameters({
            'Conv1D1/W': conv1d_2.weight.permute([2, 1, 0]).detach().numpy(),
            'Conv1D1/b': conv1d_2.bias.detach().numpy()
        })

        x_torch = torch.from_numpy(x).permute([0, 2, 1]).float()
        out1 = conv1d_2(conv1d_1(x_torch))
        out1 = out1.permute([0, 2, 1])
        out2 = myconv1d_2(myconv1d_1(x))
        np_test.assert_allclose(out1.detach().numpy(), out2, atol=1e-6, rtol=1e-6)

    def test_backward(self):
        """Test backward pass."""
        b, t = 2, 6
        x = np.random.random(size=(b, t, 10))
        conv1d_1 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        conv1d_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        x_torch = torch.from_numpy(x).permute([0, 2, 1]).float()
        out1 = conv1d_2(conv1d_1(x_torch))
        out1 = out1.permute([0, 2, 1])

        y = np.array([0., 0, 1., 0., 0.])[None, None].repeat(b, axis=0).repeat(t, axis=1)
        loss = nn.CrossEntropyLoss(reduction='sum')(
            out1.reshape([-1, out1.shape[-1]]),
            torch.from_numpy(y.copy()).reshape([-1, out1.shape[-1]])
        )
        loss.backward()
        reference_gradients = {
            'Conv1D/W': conv1d_1.weight.grad.permute([2, 1, 0]),
            'Conv1D/b': conv1d_1.bias.grad,
            'Conv1D1/W': conv1d_2.weight.grad.permute([2, 1, 0]),
            'Conv1D1/b': conv1d_2.bias.grad,
        }

        myconv1d_1 = convolution.Conv1D(10, 20)
        myconv1d_2 = convolution.Conv1D(20, 5, name='Conv1D1')
        myconv1d_1.update_parameters({
            'Conv1D/W': conv1d_1.weight.permute([2, 1, 0]).detach().numpy(),
            'Conv1D/b': conv1d_1.bias.detach().numpy()
        })
        myconv1d_2.update_parameters({
            'Conv1D1/W': conv1d_2.weight.permute([2, 1, 0]).detach().numpy(),
            'Conv1D1/b': conv1d_2.bias.detach().numpy()
        })

        output = optimizers.loss_and_grads(
            x,
            y,
            loss_fn=losses.CrossEntropy(reduction='sum', dim=-1),
            layers=[myconv1d_1, myconv1d_2]
        )

        for k, v in reference_gradients.items():
            np_test.assert_allclose(
                v.detach().numpy(), output.gradients[k], atol=1e-5, rtol=1e-5
            )


class Conv2DTest(unittest.TestCase):
    """Test for 2d convolutional network."""

    def test_forward(self):
        """Tests forward pass."""
        other_conv = nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=3,
            padding=0
        )
        other_conv2 = nn.Conv2d(
            in_channels=20,
            out_channels=5,
            kernel_size=3,
            padding=0
        )

        b, h, w, c = 2, 5, 5, 10
        image = np.random.random(size=(b, h, w, c))
        pytorch_image = torch.from_numpy(image).permute([0, 3, 1, 2]).float()
        out1 = other_conv2(other_conv(pytorch_image)).permute([0, 2, 3, 1])
        myconv = convolution.Conv2D(10, 20, 3, padding=0)
        myconv.update_parameters({
            'Conv2D/W': other_conv.weight.permute([2, 3, 1, 0]).detach().numpy(),
            'Conv2D/b': other_conv.bias.detach().numpy(),
        })
        myconv2 = convolution.Conv2D(20, 5, 3, padding=0, name='Conv2D1')
        myconv2.update_parameters({
            'Conv2D1/W': other_conv2.weight.permute([2, 3, 1, 0]).detach().numpy(),
            'Conv2D1/b': other_conv2.bias.detach().numpy(),
        })

        out2 = myconv2(myconv(image))
        np_test.assert_allclose(out1.detach().numpy(), out2, atol=1e-6, rtol=1e-6)

    def test_backward(self):
        """Test backward pass."""
        other_conv = nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=3,
            padding=0
        )
        other_conv2 = nn.Conv2d(
            in_channels=20,
            out_channels=5,
            kernel_size=3,
            padding=0
        )

        image = np.random.random(size=(2, 5, 5, 10))
        pytorch_image = torch.from_numpy(image).permute([0, 3, 1, 2]).float()
        out1 = other_conv2(other_conv(pytorch_image)).permute([0, 2, 3, 1])
        y = np.array([0., 0, 1., 0., 0.])[None, None, None].repeat(
            2, axis=0).repeat(out1.shape[1], axis=1).repeat(out1.shape[2], axis=2)
        loss = nn.CrossEntropyLoss(reduction='sum')(
            out1.reshape([-1, out1.shape[-1]]),
            torch.from_numpy(y.copy()).reshape([-1, out1.shape[-1]])
        )
        loss.backward()
        reference_gradients = {
            'Conv2D/W': other_conv.weight.grad.permute([2, 3, 1, 0]).detach().numpy(),
            'Conv2D/b': other_conv.bias.grad.detach().numpy(),
            'Conv2D1/W': other_conv2.weight.grad.permute([2, 3, 1, 0]).detach().numpy(),
            'Conv2D1/b': other_conv2.bias.grad.detach().numpy(),
        }

        myconv = convolution.Conv2D(10, 20, 3, padding=0)
        myconv.update_parameters({
            'Conv2D/W': other_conv.weight.permute([2, 3, 1, 0]).detach().numpy(),
            'Conv2D/b': other_conv.bias.detach().numpy(),
        })
        myconv2 = convolution.Conv2D(20, 5, 3, padding=0, name='Conv2D1')
        myconv2.update_parameters({
            'Conv2D1/W': other_conv2.weight.permute([2, 3, 1, 0]).detach().numpy(),
            'Conv2D1/b': other_conv2.bias.detach().numpy(),
        })

        output = optimizers.loss_and_grads(
            image,
            y,
            loss_fn=losses.CrossEntropy(reduction='sum', dim=-1),
            layers=[myconv, myconv2]
        )

        for k, v in reference_gradients.items():
            np_test.assert_allclose(v, output.gradients[k], atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
