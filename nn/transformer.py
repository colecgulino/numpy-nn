"""Transformer network module."""

import numpy as np

from nn import activations
from nn import attention
from nn import dense
from nn import layer
from nn import normalization


class FFN(layer.Layer):
    """Feed Forward Network layer for a transformer."""

    def __init__(
            self,
            embed_dim: int,
            use_bias: bool = True,
            name: str = 'FFN'
    ) -> None:
        super().__init__(name)
        self.embed_dim = embed_dim
        self.use_bias = use_bias

        self.ffn1 = dense.Dense(embed_dim, embed_dim * 4, name=f'{name}/Dense1')
        self.ffn2 = dense.Dense(embed_dim * 4, embed_dim, name=f'{name}/Dense2')
        self.act = activations.ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Implements a simple FFN (ffn1 -> act -> ffn2)."""
        h1 = self.ffn1(x)
        a1 = self.act(h1)
        return self.ffn2(self.act(self.ffn1(x))), {'h1': h1, 'a1': a1}

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards pass of the FFN network. See Dense for more details."""
        h1, a1 = cache['h1'], cache['a1']
        backwards_gradient = self.ffn2.backward(a1, {}, backwards_gradient, gradients)
        backwards_gradient = self.act.backward(h1, {}, backwards_gradient, gradients)
        return self.ffn1.backward(x, {}, backwards_gradient, gradients)


class Block(layer.Layer):
    """Single transformer block with prenorm."""

    def __init__(
            self,
            embed_dim: int,
            use_bias: bool = True,
            name: str = 'Block'
    ) -> None:
        super().__init__(name)
        self.attn = attention.MaskedSelfAttention(
            embed_dim=embed_dim,
            use_bias=use_bias,
            name=f'{name}/Attn'
        )
        self.ln_attn = normalization.LayerNorm(
            embed_dim, name=f'{name}/LNAttn'
        )
        self.ffn = FFN(
            embed_dim=embed_dim,
            use_bias=use_bias,
            name=f'{name}/FFN'
        )
        self.ln_ffn = normalization.LayerNorm(
            embed_dim, name=f'{name}/LNFFN'
        )

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Simple forward pass of the Transformer Block.
        
        Implements:
            x = x + self.attn(self.ln_attn(x))
            y = x + self.ffn(self.ln_ffn(x))
        """
        attn_in = self.ln_attn(x)
        attn_out, attn_cache = self.attn.forward(attn_in)
        ln_ffn_in = x + attn_out
        ffn_in = self.ln_ffn(ln_ffn_in)
        ffn_out, ffn_cache = self.ffn.forward(ffn_in)
        out = ln_ffn_in + ffn_out
        cache = {
            'attn_in': attn_in,
            'ln_ffn_in': ln_ffn_in,
            'ffn_in': ffn_in,
            'ffn_cache': ffn_cache,
            'attn_cache': attn_cache
        }
        return out, cache

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        attn_in, ln_ffn_in, ffn_in, = (
            cache['attn_in'], cache['ln_ffn_in'], cache['ffn_in']
        )

        ffn_backwards = self.ffn.backward(
            ffn_in, cache['ffn_cache'], backwards_gradient, gradients
        )
        ln_ffn_backwards = self.ln_ffn.backward(
            ln_ffn_in, {}, ffn_backwards, gradients
        )
        attn_backwards = self.attn.backward(
            attn_in,
            cache['attn_cache'],
            ln_ffn_backwards + backwards_gradient,
            gradients
        )
        ln_attn_backwards = self.ln_attn.backward(
            x, {}, attn_backwards, gradients
        )
        return backwards_gradient + ln_attn_backwards + ln_ffn_backwards
