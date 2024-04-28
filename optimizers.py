"""Optimizer implementation."""

import abc
import dataclasses
import typing

import numpy as np

import layer
import loss_base

Parameters = dict[str, np.ndarray]


@dataclasses.dataclass
class BackwardPassOutput:
    """Output of the `loss_and_grads` function.
    
    Args:
        loss: Scalar loss value.
        output: Output of the final layer of the model.
        gradients: Accumulated gradients for all layers.
    """
    loss: float
    output: np.ndarray
    gradients: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict
    )


def loss_and_grads(
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: loss_base.Loss,
        layers: list[layer.Layer],
        layers_kwargs: list[dict[str, typing.Any]] | None = None
) -> BackwardPassOutput:
    """Runs loss and calculates the gradients for sequential layers.
    
    Args:
        x: Input to the first layer in `layers`.
        y: Target of the loss function.
        loss_fn: Loss function to optimize for.
        layers: List of layers in sequential order to run for the network.
        layers_kwargs: Optional list of keyword arguments for each layer.
    
    Returns:
        An instance of `BackwardPassOutput`.
    """
    if layers_kwargs:
        assert len(layers_kwargs) == len(layers)
        layer_it = zip(layers, layers_kwargs)
    else:
        layer_it = zip(layers, [{} for _ in range(len(layers))])
    # Store the activations and caches for all of the layers.
    inputs, caches = [], []
    for layer, layer_kwargs in layer_it:
        inputs.append(x)
        x, cache = layer.forward(x, **layer_kwargs)
        caches.append(cache)
    # The last output is the network output.
    output = x

    # Now run the backward pass of the network and aggregate the
    # gradients.
    loss = loss_fn(output, y).item()
    backwards_gradient = loss_fn.backward(output, y)
    gradients = {}
    for input, cache, layer in zip(inputs[::-1], caches[::-1], layers[::-1]):
        backwards_gradient = layer.backward(
            input, cache, backwards_gradient, gradients
        )
    return BackwardPassOutput(loss=loss, gradients=gradients, output=output)


class Optimizer(abc.ABC):
    """Base optimizer class."""

    def __init__(
            self,
            parameters: Parameters,
            loss: loss_base.Loss,
            name='Optimizer'
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.loss = loss

    @abc.abstractmethod
    def step(
            self,
            x: np.ndarray,
            y: np.ndarray,
            layers: list[layer.Layer],
            layers_kwargs: list[dict[str, typing.Any]] | None = None
    ) -> BackwardPassOutput:
        """Step the optimizer one step.
        
        Args:
            x: Input to the layers of shape [..., in_dim].
            y: Target of the loss function of shape [..., out_dim].
            layers: Layers that make up the computation to optimize.
            layers_kwargs: Optional list of keyword arguments for each layer.

        Returns:
            An instance of `BackwardPassOutput`.
        """


class SGD(Optimizer):
    def __init__(
            self,
            lr: float,
            parameters: Parameters,
            loss: loss_base.Loss,
            name='SGD'
    ) -> None:
        super().__init__(parameters, loss, name)
        self.lr = lr

    def step(
            self,
            x: np.ndarray,
            y: np.ndarray,
            layers: list[layer.Layer],
            layers_kwargs: list[dict[str, typing.Any]] | None = None
    ) -> BackwardPassOutput:
        """Steps the optimization one step.

        Implements this update step:
            theta = theta - lr * dtheta

        Args:
            x: Input to the layers of shape [..., in_dim].
            y: Target of the loss function of shape [..., out_dim].
            layers: Layers that make up the computation to optimize.
            layers_kwargs: Optional list of keyword arguments for each layer.

        Returns:
            An instance of `BackwardPassOutput`.
        """
        # Run the backward pass of the network.
        backward_output = loss_and_grads(x, y, self.loss, layers, layers_kwargs)
        # Update the parameters.
        for k in self.parameters:
            self.parameters[k] = self.parameters[k] - self.lr * backward_output.gradients[k]
        for layer in layers:
            layer.update_parameters(self.parameters)
        return backward_output


class Momentum(Optimizer):
    def __init__(
            self,
            lr: float,
            parameters: Parameters,
            loss: loss_base.Loss,
            beta: float = 0.9,
            name='SGD'
    ) -> None:
        super().__init__(parameters, loss, name)
        self.lr = lr
        self.beta = beta
        self.momentum = {
            k: np.zeros_like(v) for k, v in self.parameters.items()
        }

    def step(
            self,
            x: np.ndarray,
            y: np.ndarray,
            layers: list[layer.Layer],
            layers_kwargs: list[dict[str, typing.Any]] | None = None
    ) -> BackwardPassOutput:
        """Steps the optimization one step.

        Implements this update step:
            theta = theta - lr * dtheta

        Args:
            x: Input to the layers of shape [..., in_dim].
            y: Target of the loss function of shape [..., out_dim].
            layers: Layers that make up the computation to optimize.
            layers_kwargs: Optional list of keyword arguments for each layer.

        Returns:
            An instance of `BackwardPassOutput`.
        """
        # Run the backward pass of the network.
        backward_output = loss_and_grads(x, y, self.loss, layers, layers_kwargs)
        # Update the parameters.
        for k in self.parameters:
            self.momentum[k] = self.beta * self.momentum[k] + (1 - self.beta) * backward_output.gradients[k]
            self.parameters[k] = self.parameters[k] - self.lr * self.momentum[k]
        for layer in layers:
            layer.update_parameters(self.parameters)
        return backward_output


class Adam(Optimizer):
    def __init__(
            self,
            lr: float,
            parameters: Parameters,
            loss: loss_base.Loss,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-6,
            name='SGD'
    ) -> None:
        super().__init__(parameters, loss, name)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {
            k: np.zeros_like(v) for k, v in self.parameters.items()
        }
        self.v = {
            k: np.zeros_like(v) for k, v in self.parameters.items()
        }
        self.t = 0

    def step(
            self,
            x: np.ndarray,
            y: np.ndarray,
            layers: list[layer.Layer],
            layers_kwargs: list[dict[str, typing.Any]] | None = None
    ) -> BackwardPassOutput:
        """Steps the optimization one step.

        Implements this update step:
            theta = theta - lr * dtheta

        Args:
            x: Input to the layers of shape [..., in_dim].
            y: Target of the loss function of shape [..., out_dim].
            layers: Layers that make up the computation to optimize.
            layers_kwargs: Optional list of keyword arguments for each layer.

        Returns:
            An instance of `BackwardPassOutput`.
        """
        # Run the backward pass of the network.
        backward_output = loss_and_grads(x, y, self.loss, layers, layers_kwargs)
        gradients = backward_output.gradients
        # Update the parameters.
        for k in self.parameters:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * gradients[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * np.square(gradients[k])
            m_hat = self.m[k] / (1 - self.beta1**(self.t + 1))
            v_hat = self.v[k] / (1 - self.beta2**(self.t + 1))
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            self.parameters[k] = self.parameters[k] - self.lr * update
        for layer in layers:
            layer.update_parameters(self.parameters)
        self.t += 1
        return backward_output
