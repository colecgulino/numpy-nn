"""Layers for activations in neural networks."""

import numpy as np

from nn import layer


class Sigmoid(layer.Layer):
    """Sigmoid layer for activations.
    
    Implements 1 / (1 + e^-x) in the forward layer and
    sigmoid(x) * (1 - sigmoid(x)) for the backward pass.
    """

    def __init__(self, name: str = 'Sigmoid') -> None:
        super().__init__(name)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Implements elementwise sigmoid fn 1 / (1 + e^-x)."""
        return 1. / (1 + np.exp(-x)), {}

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards pass of the sigmoid function.

        Implements: sigmoid(x) * (1 - sigmoid(x))

        Derivation:
            d/dx (1 / (1 + e^-x)) = [d(1)/dx * (1 + e^-x) - d(1 + e^-x)/dx * 1] / (1 + e^-x)^2
                                  = - d(1 + e^-x)/dx / (1 + e^-x)^2
                                  = [- d(1)/dx - d(e^-x)/dx] / (1 + e^-x)^2
                                  = [-e^-x * d(-x)/dx] / (1 + e^-x)^2
                                  = e^-x / (1 + e^-x)^2
                                  = [1 / (1 + e^-x)][e^-x / (1 + e^-x)]
                                  = [1 / (1 + e^-x)][(e^-x + 1 - 1) / (1 + e^-x)]
                                  = sigmoid(x)[(1 + e^-x)/(1 + e^-x) - 1 / (1 + e^-x)]
                                  = sigmoidx(x)[1 - sigmoid(x)]
        Args:
            x: Input of the network of shape [..., in_dim].
            cache: Cache from the forward function to pass into the backward pass.
            backwards_gradient: Gradient from the previous layer that represents
                dout[prev_layer] / dout[this_layer].
            gradients: Mutable dictionary which will not be updated as there are
                no parameters of this function.
        
        Returns:
            Gradient to pass onto the previous layer of shape [..., in_dim].
        """
        del gradients, cache
        y = self.forward(x)[0]
        return backwards_gradient * y * (1. - y)


class ReLU(layer.Layer):
    """Implements rectified linear units.
    
    y(x) = { x  if x > 0
           { 0  else
    """

    def __init__(self, name: str = 'ReLU') -> None:
        super().__init__(name)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Implements forward pass of ReLU."""
        return (x > 0.).astype(np.float32) * x, {}

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards pass of the ReLU function.

        Implements: 
            dy(x) / dx = { 1  if x > 0
                         { 0  else

        Args:
            x: Input of the network of shape [..., in_dim].
            cache: Cache from the forward function to pass into the backward pass.
            backwards_gradient: Gradient from the previous layer that represents
                dout[prev_layer] / dout[this_layer].
            gradients: Mutable dictionary which will not be updated as there are
                no parameters of this function.
        
        Returns:
            Gradient to pass onto the previous layer of shape [..., in_dim].
        """
        del gradients, cache
        return backwards_gradient * (x > 0.).astype(np.float32)


class Softmax(layer.Layer):
    """Implements softmax layer.
    
    y(x_i) = e^x_i / sum_j e^x_j
    """

    def __init__(self, dim: int = 0, name: str = 'Softmax') -> None:
        super().__init__(name)
        self.dim = dim

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Implements softmax function y(x_i) = e^x_i / sum_j e^x_j"""
        max_value = np.max(x, axis=self.dim, keepdims=True)
        exp = np.exp(x - max_value)
        exp_sum = np.sum(exp, axis=self.dim, keepdims=True)
        return exp / exp_sum, {}

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards pass of the Softmax function.

        Softmax derivative is actually a Jacobian because the output of a single
        value in the matrix is a function of many values.
            d y(x_i) / dx_j = { i = j -softmax(x)softmax(x)
                              { else softmax(x)(1 - softmax(x))

        Derivation:
            d y(x_i) / dx_j = d(e^x_i / sum_k e^x_k) / dx_i
                            = [(e^x_i)'(sum_k e^x_k) - (e^x_i)(sum_k e^x_k)'] / [(sum_k e^x_k)]^2
                            = [(e^x_i)'(sum_k e^x_k) - (e^x_i)(e^x_j)] / [(sum_k e^x_k)]^2
                if i != j
                            = [(-e^x_i)(e^x_j)] / [sum_k e^x_k] ^ 2
                            = [-e^x_i / sum_k e^x_k][e^x_j / sum_k e^x_k]
                            = -softmax(x_i)softmax(x_j)
                if i == j
                            = [(e^x_i)sum_k e^x_k - (e^x_i)^2] / [(sum_k e^x_k)]^2
                            = [e^x_i(sum_k e^x_k - e^x_i)] / [(sum_k e^x_k)]^2
                            = [e^x_i / sum_k e^x_k][(sum_k e^x_k - e^x_i) / sum_k e^x_k]
                            = softmax(x_i)[(sum_k e^x_k / sum_k e^x_k) - (e^x_i / sum_k e^x_k)]
                            = softmax(x_i)(1 - softmax(x_i))

        Args:
            x: Input of the network of shape [..., in_dim].
            cache: Cache from the forward function to pass into the backward pass.
            backwards_gradient: Gradient from the previous layer that represents
                dout[prev_layer] / dout[this_layer].
            gradients: Mutable dictionary which will not be updated as there are
                no parameters of this function.
        
        Returns:
            Gradient to pass onto the previous layer of shape [..., in_dim, in_dim].
        """
        del gradients, cache
        batch_dims = x.shape[:-1]
        eye = np.eye(x.shape[-1])
        for batch_dim in reversed(batch_dims):
            eye = eye[None].repeat(batch_dim, axis=0)
        # Shape: [..., D].
        scores = self.forward(x)[0]
        # Shape: [..., D, D].
        softmax_jacobian = scores[..., None] * (eye - scores[..., None])
        return (softmax_jacobian @ backwards_gradient[..., None]).squeeze(-1)


class Tanh(layer.Layer):
    """Implements tanh function tanh(x) = sinh(x) / cosh(x)"""

    def __init__(self, name: str = 'Tanh') -> None:
        super().__init__(name)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Implements forward pass of the tanh(x) function.
        
        tanh(x) = sinh(x) / cosh(x)
        sinh(x) = (1/2) * (e^x - e^(-x))
        cosh(x) = (1/2) * (e^x + e^(-x))
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
                = (e^(2x) - 1) / (e^(2x) + 1)

        Ags:
            x: Input to the function of shape [..., in_dim]
        
        Returns:
            tanh(x) function output with values in range [-1, 1].
        """
        exp_2x = np.exp(2 * x)
        return (exp_2x - 1) / (exp_2x + 1), {}

    def backward(
            self,
            x: np.ndarray,
            cache: layer.Cache,
            backwards_gradient: np.ndarray,
            gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards pass of the tanh(x) function.

        Derivation:
            dtanh(x) / dx = (e^(2x) + 1)^-1 (e^(2x) - 1)/dx + 
                            (e^(2x) - 1) (e^(2x) + 1)^-1/dx
                          = 2(e^(2x) + 1)^-1 e^(2x) -
                            2(e^(2x) - 1) (e^(2x) + 1)^-2 e^(2x)
                          = 2e^(2x) (e^(2x) + 1)^-1(1 - (e^(2x) - 1) / (e^(2x) + 1))
                          = 2e^(2x) (e^(2x) + 1)^-1 (1 - tanh(x))

        Args:
            x: Input of the network of shape [..., in_dim].
            cache: Cache from the forward function to pass into the backward pass.
            backwards_gradient: Gradient from the previous layer that represents
                dout[prev_layer] / dout[this_layer].
            gradients: Mutable dictionary which will not be updated as there are
                no parameters of this function.
        
        Returns:
            Gradient to pass onto the previous layer of shape [..., in_dim].
        """
        del gradients, cache
        tanhx, _ = self.forward(x)
        return backwards_gradient * (1 - tanhx**2)
