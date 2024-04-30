"""Regularization layers."""

import numpy as np

from nn import layer


class Dropout(layer.Layer):
    """Dropout layer."""

    def __init__(self, p: float, name: str = 'Dropout') -> None:
        super().__init__(name)
        self.p = p
        self.set_parameter('mask', None)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Forward pass of random dropout."""
        mask = (np.random.random(x.shape) > self.p).astype(np.float32)
        return x * mask, {'mask': mask}

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backward pass of random dropout."""
        del x, gradients
        return backwards_gradient * cache['mask']
