"""Calculates multihead attention for an input sequence."""
import math

import numpy as np

import activations
import dense
import layer


def get_causal_mask(x: np.ndarray) -> np.ndarray:
    """Gets causal mask for a sequence."""
    batch, time, _ = x.shape
    # Shape: [T, T].
    mask = np.triu(np.ones((time, time))) - np.eye(time)
    # Shape: [B, T, T].
    return mask[None].repeat(batch, axis=0).astype(np.bool_)


class MaskedSelfAttention(layer.Layer):
    """Masked self attention layer for a transformer."""

    def __init__(self, embed_dim: int, use_bias: bool = True, name='MSA') -> None:
        super().__init__(name)
        self.embed_dim = embed_dim

        self.q_proj = dense.Dense(embed_dim, embed_dim, name=f'{name}/DenseQ', use_bias=use_bias)
        self.k_proj = dense.Dense(embed_dim, embed_dim, name=f'{name}/DenseK', use_bias=use_bias)
        self.v_proj = dense.Dense(embed_dim, embed_dim, name=f'{name}/DenseV', use_bias=use_bias)
        self.output_proj = dense.Dense(embed_dim, embed_dim, name=f'{name}/DenseO')
        self.softmax = activations.Softmax(dim=-1)

    def forward(
            self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, layer.Cache]:
        """Runs forward pass of MH Self-Attention.
        
        Q = Wq * x; K = Wk * x; V = Wv * x
        scores = softmax(Q @ K / scale)
        O = Wo * scores @ K

        Args:
            x: Input to the module of shape [B, T, C].
            mask: Attention mask of shape [B, T, T] of dtype bool.
        
        Returns:
            Output of the attention module of shape: [B, T, C].
        """
        # Shape: [B, T, C].
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Shape: [B, T, T].
        scale = 1. / math.sqrt(k.shape[-1])
        attn = q @ k.transpose([0, 2, 1]) * scale
        if mask is not None:
            # Shape: [B, T, T].
            attn = np.where(mask, -float("inf"), attn)
        scores = self.softmax(attn)

        # Shape: [B, T, C // n_heads].
        output = scores @ v
        # Shape: [B, T, C].
        y = self.output_proj(output)
        cache = {
            'q': q, 'k': k, 'v': v, 'attn': attn, 'scores': scores,
            'output': output, 'mask': mask
        }
        return y, cache

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards pass of the multi-headed attention function.

        Args:
            x: Input of the network of shape [B, T, C].
            cache: Cache from the forward function to pass into the backward pass.
            backwards_gradient: Gradient from the previous layer that represents
                dout[prev_layer] / dout[this_layer] of shape [B, T, C].
            gradients: Mutable dictionary which will not be updated as there are
                no parameters of this function.
        
        Returns:
            Gradient to pass onto the previous layer of shape [B, T, C] and
                calculating.
        """
        # Find the gradient w.r.t. the output projection.
        # Shape: [B, T, C].
        dy_doutput = self.output_proj.backward(
            cache['output'], {}, backwards_gradient, gradients
        )

        # Get the gradients w.r.t. V
        # Shape: [B,, T, C].
        # doutput / dv -> Shape: [B,, T, C].
        dy_dv = cache['scores'] @ dy_doutput
        v_backwards = self.v_proj.backward(x, {}, dy_dv, gradients)
        # Shape: [B, T, C].

        # Get the gradients w.r.t. attention.
        # Shape: [B,, T, T].
        dy_dscores = backwards_gradient @ cache['v'].transpose([0, 2, 1])
        # Shape: [B, T, T].
        dy_dattn = self.softmax.backward(cache['attn'], {}, dy_dscores, {})
        mask = cache['mask']
        if mask is not None:
            dy_dattn = dy_dattn * mask.astype(np.float32)

        # Get the gradients w.r.t. Q and K.
        # Shape: [B,, T, C].
        k_backwards = self.k_proj.backward(x, {}, dy_dattn @ cache['q'], gradients)
        q_backwards = self.q_proj.backward(x, {}, dy_dattn @ cache['k'], gradients)
        return k_backwards + q_backwards + v_backwards
