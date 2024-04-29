"""Tests transformer layers."""

import math
import unittest

from numpy import testing as np_test
import torch
from torch import nn

import losses
import optimizers
import transformer


NUM_BLOCKS = 5


class MSATorch(torch.nn.Module):
    """Test self attention layer."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.output_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Test msa forward pass."""
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


class PTBlock(nn.Module):
    """Test transformer block."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.attn = MSATorch(embed_dim)
        self.ln_attn = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Linear(embed_dim, embed_dim * 4)
        self.ffn2 = nn.Linear(embed_dim * 4, embed_dim)
        self.act = nn.ReLU()
        self.ln_ffn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Forward pass of test transformer block."""
        temp = x + self.attn(self.ln_attn(x))
        return temp + self.ffn2(
            self.act(self.ffn1(self.ln_ffn(temp)))
        )


class TestTransformer(unittest.TestCase):
    """Test tranfsormer network."""

    def test_forward(self):
        """Test forward pass."""
        x = torch.ones((5, 10, 10))

        pt_blocks = []
        parameters = {}
        for i in range(NUM_BLOCKS):
            pt_blocks.append(PTBlock(10))
            parameters.update({
                f'Block{i}/LNFFN/scale': pt_blocks[-1].ln_ffn.weight,
                f'Block{i}/LNFFN/shift': pt_blocks[-1].ln_ffn.bias,
                f'Block{i}/FFN/Dense1/W': pt_blocks[-1].ffn1.weight.T,
                f'Block{i}/FFN/Dense1/b': pt_blocks[-1].ffn1.bias,
                f'Block{i}/FFN/Dense2/W': pt_blocks[-1].ffn2.weight.T,
                f'Block{i}/FFN/Dense2/b': pt_blocks[-1].ffn2.bias,
                f'Block{i}/Attn/DenseQ/W': pt_blocks[-1].attn.q_proj.weight.T,
                f'Block{i}/Attn/DenseQ/b': pt_blocks[-1].attn.q_proj.bias,
                f'Block{i}/Attn/DenseK/W': pt_blocks[-1].attn.k_proj.weight.T,
                f'Block{i}/Attn/DenseK/b': pt_blocks[-1].attn.k_proj.bias,
                f'Block{i}/Attn/DenseV/W': pt_blocks[-1].attn.v_proj.weight.T,
                f'Block{i}/Attn/DenseV/b': pt_blocks[-1].attn.v_proj.bias,
                f'Block{i}/Attn/DenseO/W': pt_blocks[-1].attn.output_proj.weight.T,
                f'Block{i}/Attn/DenseO/b': pt_blocks[-1].attn.output_proj.bias,
                f'Block{i}/LNAttn/scale': pt_blocks[-1].ln_attn.weight,
                f'Block{i}/LNAttn/shift': pt_blocks[-1].ln_attn.bias,
            })
        parameters = {k: v.detach().numpy() for k, v in parameters.items()}

        my_blocks = []
        for i in range(NUM_BLOCKS):
            my_blocks.append(transformer.Block(10, 1, name=f'Block{i}'))
            my_blocks[-1].update_parameters(parameters)

        y = x.detach().numpy()
        for block in my_blocks:
            y = block(y)
        y2 = nn.Sequential(*pt_blocks)(x).detach().numpy()
        np_test.assert_allclose(y, y2, atol=1e-5, rtol=1e-5)

    def test_backward(self):
        """Test backward pass."""
        x = torch.ones((5, 10, 10))

        pt_blocks = []
        parameters = {}
        for i in range(NUM_BLOCKS):
            pt_blocks.append(PTBlock(10))
            parameters.update({
                f'Block{i}/LNFFN/scale': pt_blocks[-1].ln_ffn.weight,
                f'Block{i}/LNFFN/shift': pt_blocks[-1].ln_ffn.bias,
                f'Block{i}/FFN/Dense1/W': pt_blocks[-1].ffn1.weight.T,
                f'Block{i}/FFN/Dense1/b': pt_blocks[-1].ffn1.bias,
                f'Block{i}/FFN/Dense2/W': pt_blocks[-1].ffn2.weight.T,
                f'Block{i}/FFN/Dense2/b': pt_blocks[-1].ffn2.bias,
                f'Block{i}/Attn/DenseQ/W': pt_blocks[-1].attn.q_proj.weight.T,
                f'Block{i}/Attn/DenseQ/b': pt_blocks[-1].attn.q_proj.bias,
                f'Block{i}/Attn/DenseK/W': pt_blocks[-1].attn.k_proj.weight.T,
                f'Block{i}/Attn/DenseK/b': pt_blocks[-1].attn.k_proj.bias,
                f'Block{i}/Attn/DenseV/W': pt_blocks[-1].attn.v_proj.weight.T,
                f'Block{i}/Attn/DenseV/b': pt_blocks[-1].attn.v_proj.bias,
                f'Block{i}/Attn/DenseO/W': pt_blocks[-1].attn.output_proj.weight.T,
                f'Block{i}/Attn/DenseO/b': pt_blocks[-1].attn.output_proj.bias,
                f'Block{i}/LNAttn/scale': pt_blocks[-1].ln_attn.weight,
                f'Block{i}/LNAttn/shift': pt_blocks[-1].ln_attn.bias,
            })
        parameters = {k: v.detach().numpy() for k, v in parameters.items()}

        my_blocks = []
        for i in range(NUM_BLOCKS):
            my_blocks.append(transformer.Block(10, 1, name=f'Block{i}'))
            my_blocks[-1].update_parameters(parameters)

        y = x.detach().numpy()
        for block in my_blocks:
            y = block(y)
        y2 = nn.Sequential(*pt_blocks)(x)

        y_hat = torch.tensor(
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
        )[None, None].repeat(5, 10, 1)
        loss = nn.CrossEntropyLoss()(
            y2.contiguous().view(-1, 10), y_hat.contiguous().view(-1, 10)
        )
        loss.backward()

        reference_gradients = {}
        for i in range(NUM_BLOCKS):
            pt_blocks.append(PTBlock(10))
            reference_gradients.update({
                f'Block{i}/LNFFN/scale': pt_blocks[i].ln_ffn.weight.grad,
                f'Block{i}/LNFFN/shift': pt_blocks[i].ln_ffn.bias.grad,
                f'Block{i}/FFN/Dense1/W': pt_blocks[i].ffn1.weight.grad.T,
                f'Block{i}/FFN/Dense1/b': pt_blocks[i].ffn1.bias.grad,
                f'Block{i}/FFN/Dense2/W': pt_blocks[i].ffn2.weight.grad.T,
                f'Block{i}/FFN/Dense2/b': pt_blocks[i].ffn2.bias.grad,
                f'Block{i}/Attn/DenseQ/W': pt_blocks[i].attn.q_proj.weight.grad.T,
                f'Block{i}/Attn/DenseQ/b': pt_blocks[i].attn.q_proj.bias.grad,
                f'Block{i}/Attn/DenseK/W': pt_blocks[i].attn.k_proj.weight.grad.T,
                f'Block{i}/Attn/DenseK/b': pt_blocks[i].attn.k_proj.bias.grad,
                f'Block{i}/Attn/DenseV/W': pt_blocks[i].attn.v_proj.weight.grad.T,
                f'Block{i}/Attn/DenseV/b': pt_blocks[i].attn.v_proj.bias.grad,
                f'Block{i}/Attn/DenseO/W': pt_blocks[i].attn.output_proj.weight.grad.T,
                f'Block{i}/Attn/DenseO/b': pt_blocks[i].attn.output_proj.bias.grad,
                f'Block{i}/LNAttn/scale': pt_blocks[i].ln_attn.weight.grad,
                f'Block{i}/LNAttn/shift': pt_blocks[i].ln_attn.bias.grad,
            })
        reference_gradients = {k: v.detach().numpy() for k, v in reference_gradients.items()}

        activations, caches = [x.detach().numpy()], []
        for block in my_blocks:
            activation, cache = block.forward(activations[-1])
            activations.append(activation)
            caches.append(cache)

        loss_fn = losses.CrossEntropy(dim=-1)
        output = optimizers.loss_and_grads(
            x.detach().numpy(), y_hat.detach().numpy(), loss_fn, my_blocks
        )

        for k, v in reference_gradients.items():
            np_test.assert_allclose(v, output.gradients[k], atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
 