"""Implementation of a Dense Neural Network Layer."""

import functools

import numpy as np

import initializers
import layer


class Dense(layer.Layer):
    """Calculates a dense linear layer."""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            use_bias: bool = True,
            name: str = 'Dense'
    ) -> None:
        """Constructs the Dense Layer.
        
        Args:
            in_dim: Dimension of the input of shape [..., in_dim].
            out_dim: Dimension of the output of shape [..., out_dim].
            use_bias: Whether or not to use an optional bias parameter.
            name: Name of the network layer. Should be unique.
        """
        super().__init__(name)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        self.set_parameter(
            'W',
            initializers.xavier_uniform(in_dim, out_dim, (in_dim, out_dim))
        )
        if use_bias:
            self.set_parameter('b', np.zeros((out_dim)))

    def parameters(self) -> dict[str, np.ndarray]:
        return self._parameters

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Forward pass of a dense network.
        
        Implements:
            y = Wx + b
        
        Args:
            x: Input tensor of shape [..., in_dim].
        
        Returns:
            Output tensor of shape [..., out_dim].
        """
        x = x @ self.get_parameter('W')
        if self.use_bias:
            x = x + self.get_parameter('b')
        return x, {}

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Runs the backwards pass over the dense network.

            dy/dW = x.T @ backwards_gradient
            dy/db = (backwards_gradient).T @ 1
        
        Args:
            x: Input of the network of shape [..., in_dim].
            cache: Cache from the forward function to pass into the backward pass.
            backwards_gradient: Gradient from the previous layer that represents
                dout[prev_layer] / dout[this_layer].
            gradients: Mutable dictionary which should be updated with the
                gradients for all parameters in this layer.
        
        Returns:
            Gradient to pass onto the previous layer of shape [..., in_dim] and
                calculating dy/dx = backwards_gradient @ W.T.
        """
        del cache
        batch_dims = x.shape[:-1]
        b = functools.reduce(lambda x, y: x * y, batch_dims, 1)
        # Shape: [b, in_dim].
        x = x.reshape(b, self.in_dim)
        # Shape: [b, out_dim].
        backwards_gradient = backwards_gradient.reshape(b, self.out_dim)

        # Shape: [in_dim, out_dim].
        dL_dw = x.T @ backwards_gradient
        gradients[f'{self.name}/W'] = dL_dw
        # Shape: [out_dim].
        dL_db = backwards_gradient.sum(axis=0)
        gradients[f'{self.name}/b'] = dL_db

        # Calculate the backwards gradient.
        # Shape: [b, in_dim].
        backwards_gradient = backwards_gradient @ self.get_parameter('W').T
        # Reshape to: [*batch_dims, in_dim].
        backwards_gradient = backwards_gradient.reshape(*batch_dims, self.in_dim)
        return backwards_gradient
