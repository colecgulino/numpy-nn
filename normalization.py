"""Normalization layers."""

import functools

import numpy as np

import layer


class LayerNorm(layer.Layer):
    """Implementation of LayerNorm (https://arxiv.org/abs/1607.06450)."""

    def __init__(self, size: int, eps: float = 0.00001, name: str = 'LayerNorm') -> None:
        super().__init__(name)
        self.size = size
        self.eps = eps
        self.set_parameter('scale', np.ones((self.size)))
        self.set_parameter('shift', np.zeros((self.size)))
    
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Forward propogation for LayerNorm.
        Implements:
            y(x) = scale * [x - x.mean()] / [(sum_i [x_i - x.mean()]^2)^1/2] + shift
        
        Args:
            x: Input to the module of shape [..., size].
        
        Return:
            Normalized output of shape [..., size].
        """
        C = x.shape[-1]
        assert C == self.size
        # Compute the mean and variance along the last dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

        # Normalize the inputs
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift the normalized inputs
        gamma = self.get_parameter('scale')
        beta = self.get_parameter('shift')
        out = gamma * x_norm + beta
        return out, {}
    
    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards pass of the LayerNorm layer.

        Derivation:
            dy / dscale = backwards_gradient @ [
                x - x.mean()] / [(sum_i [x - x.mean()]^2)^1/2
            ]
            dy / dshift = backwards_gradient @ 1

        Args:
            x: Input of the network of shape [..., size].
            cache: Cache from the forward function to pass into the backward pass.
            backwards_gradient: Gradient from the previous layer that represents
                dout[prev_layer] / dout[this_layer].
            gradients: Mutable dictionary which should be updated with the
                gradients for all parameters in this layer.
        
        Returns:
            Gradient to pass onto the previous layer of shape [..., size].
        """
        del cache
        batch_dims = x.shape[:-1]
        b = functools.reduce(lambda x, y: x * y, batch_dims, 1)

        # Shape: [b, size].
        backwards_gradient = backwards_gradient.reshape(b, self.size)

        # Scale
        # Shape: [size].
        dy_dshift = backwards_gradient.sum(axis=0)
        gradients[f'{self.name}/shift'] = dy_dshift

        # Shift
        # Shape: [b, size].
        x = x.reshape(b, self.size)
        # Shape: [b, 1].
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean)**2).mean(axis=-1, keepdims=True)
        # Shape: [b, size].
        normalized = (x - mean) / np.sqrt(var + self.eps)
        # Shape: [size].
        dy_dscale = (backwards_gradient * normalized).sum(axis=0)
        gradients[f'{self.name}/scale'] = dy_dscale

        # Backwards gradient.
        N = self.size
        gamma = self.get_parameter('scale')
        sigma = np.sqrt(var + self.eps)
        dmu = 1. / N
        # Shape: [..., N].
        dsigma = (x - mean) / (N * sigma)
        # Shape: [..., N, N].
        eye = np.eye(N)[None].repeat(b, axis=0)
        jacobian = (1. / sigma[..., None]) * (eye - dmu)
        temp = ((x - mean) / sigma**2)[..., None] @ dsigma[..., None, :]
        jacobian -= temp
        jacobian *= gamma[..., None]
        dx = (jacobian @ backwards_gradient[..., None]).squeeze(-1)
        dx = dx.reshape(*batch_dims, self.size)
        return dx
