"""Test for attention in Numpy NN library."""

import math
import unittest

from numpy import testing as np_test
import torch

from nn import attention
from nn import losses


class MSATorch(torch.nn.Module):
    """Testing class for Self-Attention."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.output_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the test self attention."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scale = 1. / math.sqrt(k.shape[-1])
        attn = q @ k.transpose(-2, -1) * scale
        if mask is not None:
            attn = attn.masked_fill(mask, -float("inf"))
        scores = torch.nn.functional.softmax(attn, dim=-1)
        output = scores @ v
        return self.output_proj(output)


class TestAttention(unittest.TestCase):
    """Tests for attention layers."""

    def test_forward(self):
        """Tests forward pass of attention layer."""
        attn0_0 = MSATorch(embed_dim=10)
        attn0_1 = MSATorch(embed_dim=10)

        update_parameters = {
            'MSA/DenseQ/W': attn0_0.q_proj.weight.T,
            'MSA/DenseQ/b': attn0_0.q_proj.bias,
            'MSA/DenseK/W': attn0_0.k_proj.weight.T,
            'MSA/DenseK/b': attn0_0.k_proj.bias,
            'MSA/DenseV/W': attn0_0.v_proj.weight.T,
            'MSA/DenseV/b': attn0_0.v_proj.bias,
            'MSA/DenseO/W': attn0_0.output_proj.weight.T,
            'MSA/DenseO/b': attn0_0.output_proj.bias,
            'MSA1/DenseQ/W': attn0_1.q_proj.weight.T,
            'MSA1/DenseQ/b': attn0_1.q_proj.bias,
            'MSA1/DenseK/W': attn0_1.k_proj.weight.T,
            'MSA1/DenseK/b': attn0_1.k_proj.bias,
            'MSA1/DenseV/W': attn0_1.v_proj.weight.T,
            'MSA1/DenseV/b': attn0_1.v_proj.bias,
            'MSA1/DenseO/W': attn0_1.output_proj.weight.T,
            'MSA1/DenseO/b': attn0_1.output_proj.bias
        }
        update_parameters = {k: v.detach().numpy() for k, v in update_parameters.items()}
        attn1_0 = attention.MaskedSelfAttention(embed_dim=10)
        attn1_0.update_parameters(update_parameters)
        attn1_1 = attention.MaskedSelfAttention(embed_dim=10, name='MSA1')
        attn1_1.update_parameters(update_parameters)

        x = torch.ones((5, 20, 10))
        mask = attention.get_causal_mask(x)

        y_ref = attn0_1(
            attn0_0(x, torch.from_numpy(mask)), torch.from_numpy(mask)
        )
        y = attn1_1(attn1_0(x.detach().numpy(), mask), mask)
        np_test.assert_allclose(y, y_ref.detach().numpy(), atol=1e-6, rtol=1e-6)

    def test_backward(self):
        """Tests backward pass of attention layer."""
        attn0_0 = MSATorch(embed_dim=10)
        attn0_1 = MSATorch(embed_dim=10)

        update_parameters = {
            'MSA/DenseQ/W': attn0_0.q_proj.weight.T,
            'MSA/DenseQ/b': attn0_0.q_proj.bias,
            'MSA/DenseK/W': attn0_0.k_proj.weight.T,
            'MSA/DenseK/b': attn0_0.k_proj.bias,
            'MSA/DenseV/W': attn0_0.v_proj.weight.T,
            'MSA/DenseV/b': attn0_0.v_proj.bias,
            'MSA/DenseO/W': attn0_0.output_proj.weight.T,
            'MSA/DenseO/b': attn0_0.output_proj.bias,
            'MSA1/DenseQ/W': attn0_1.q_proj.weight.T,
            'MSA1/DenseQ/b': attn0_1.q_proj.bias,
            'MSA1/DenseK/W': attn0_1.k_proj.weight.T,
            'MSA1/DenseK/b': attn0_1.k_proj.bias,
            'MSA1/DenseV/W': attn0_1.v_proj.weight.T,
            'MSA1/DenseV/b': attn0_1.v_proj.bias,
            'MSA1/DenseO/W': attn0_1.output_proj.weight.T,
            'MSA1/DenseO/b': attn0_1.output_proj.bias
        }
        update_parameters = {k: v.detach().numpy() for k, v in update_parameters.items()}
        attn1_0 = attention.MaskedSelfAttention(embed_dim=10)
        attn1_0.update_parameters(update_parameters)
        attn1_1 = attention.MaskedSelfAttention(embed_dim=10, name='MSA1')
        attn1_1.update_parameters(update_parameters)

        x = torch.ones((5, 20, 10))
        mask = attention.get_causal_mask(x)

        y_ref = attn0_1(
            attn0_0(x, torch.from_numpy(mask)), torch.from_numpy(mask)
        )
        y = attn1_1(attn1_0(x.detach().numpy(), mask), mask)

        y_hat = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])[None, None].repeat(5, 20, 1)

        loss = torch.nn.CrossEntropyLoss()(
            y_ref.contiguous().view(-1, 10), y_hat.contiguous().view(-1, 10)
        )
        loss.backward()

        reference_gradients = {
            'MSA/DenseQ/W': attn0_0.q_proj.weight.grad.T,
            'MSA/DenseQ/b': attn0_0.q_proj.bias.grad,
            'MSA/DenseK/W': attn0_0.k_proj.weight.grad.T,
            'MSA/DenseK/b': attn0_0.k_proj.bias.grad,
            'MSA/DenseV/W': attn0_0.v_proj.weight.grad.T,
            'MSA/DenseV/b': attn0_0.v_proj.bias.grad,
            'MSA/DenseO/W': attn0_0.output_proj.weight.grad.T,
            'MSA/DenseO/b': attn0_0.output_proj.bias.grad,
            'MSA1/DenseQ/W': attn0_1.q_proj.weight.grad.T,
            'MSA1/DenseQ/b': attn0_1.q_proj.bias.grad,
            'MSA1/DenseK/W': attn0_1.k_proj.weight.grad.T,
            'MSA1/DenseK/b': attn0_1.k_proj.bias.grad,
            'MSA1/DenseV/W': attn0_1.v_proj.weight.grad.T,
            'MSA1/DenseV/b': attn0_1.v_proj.bias.grad,
            'MSA1/DenseO/W': attn0_1.output_proj.weight.grad.T,
            'MSA1/DenseO/b': attn0_1.output_proj.bias.grad
        }

        loss_fn = losses.CrossEntropy(dim=-1)
        attn1_0_out, attn1_0_cache = attn1_0.forward(x.detach().numpy(), mask)
        _, attn1_1_cache = attn1_1.forward(attn1_0_out, mask)
        gradients = {}
        backward_grad = loss_fn.backward(y, y_hat.detach().numpy())
        backward_grad = attn1_1.backward(attn1_0_out, attn1_1_cache, backward_grad, gradients)
        backward_grad = attn1_0.backward(
            x.detach().numpy(), attn1_0_cache, backward_grad, gradients
        )

        for k, v in gradients.items():
            np_test.assert_allclose(v, reference_gradients[k], rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
