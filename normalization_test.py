import math
import unittest

from numpy import testing as np_test
import torch
from torch import nn

from activations import Sigmoid
from dense import Dense
import losses
import normalization


class TestLayerNorm(unittest.TestCase):

    def test_forward(self):
        x = torch.rand((5, 10, 100))
        lnorm_torch = torch.nn.LayerNorm(x.shape[-1])
        lnorm = normalization.LayerNorm(size=x.shape[-1])
        lnorm.update_parameters({
            'LayerNorm/scale': lnorm_torch.weight.detach().numpy(),
            'LayerNorm/shift': lnorm_torch.bias.detach().numpy()
        })
        y = lnorm(x.detach().numpy())
        y_ref = lnorm_torch(x)
        np_test.assert_allclose(y, y_ref.detach().numpy(), atol=1e-6, rtol=1e-6)

    def test_backward(self):
        x = torch.ones((5, 10, 10), requires_grad=True)

        dense = Dense(10, 5)
        dense2 = Dense(5, 5, name='Dense2')
        sigmoid = Sigmoid()
        norm = normalization.LayerNorm(size=5)

        l = nn.Linear(10, 5)
        l2 = nn.Linear(5, 5)
        n = nn.LayerNorm(5)
        parameters = {
            'Dense/W': l.weight.T,
            'Dense/b': l.bias,
            'Dense2/W': l2.weight.T,
            'Dense2/b': l2.bias,
            'LayerNorm/scale': n.weight,
            'LayerNorm/shift': n.bias
        }
        parameters = {k: v.detach().numpy() for k, v in parameters.items()}
        dense.update_parameters(parameters)
        dense2.update_parameters(parameters)
        norm.update_parameters(parameters)
        y2 = l2(torch.nn.functional.sigmoid(n(l(x))))
        
        y_hat = torch.tensor([0., 1., 0., 0., 0.])[None, None].repeat(5, 10, 1)
        loss = nn.CrossEntropyLoss()(y2.contiguous().view(-1, 5), y_hat.contiguous().view(-1, 5))
        loss.backward()

        reference_gradients = {
            'Dense/W': l.weight.grad.T,
            'Dense/b': l.bias.grad,
            'Dense2/W': l2.weight.grad.T,
            'Dense2/b': l2.bias.grad,
            'LayerNorm/scale': n.weight.grad,
            'LayerNorm/shift': n.bias.grad
        }
        reference_gradients = {k: v.detach().numpy() for k, v in reference_gradients.items()}

        dense_out, dense_cache = dense.forward(x.detach().numpy())
        norm_out, norm_cache = norm.forward(dense_out)
        sigmoid_out, sigmoid_cache = sigmoid.forward(norm_out)
        y, dense2_cache = dense2.forward(sigmoid_out)

        gradients = {}
        loss_fn = losses.CrossEntropy(dim=-1)
        backward_grad = loss_fn.backward(y, y_hat.detach().numpy())
        backward_grad = dense2.backward(sigmoid_out, dense2_cache, backward_grad, gradients)
        backward_grad = sigmoid.backward(norm_out, sigmoid_cache, backward_grad, gradients)
        backward_grad = norm.backward(dense_out, norm_cache, backward_grad, gradients)
        backward_grad = dense.backward(x.detach().numpy(), dense_cache, backward_grad, gradients)

        for k, v in reference_gradients.items():
            np_test.assert_allclose(v, gradients[k], atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
 