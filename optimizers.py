"""Optimizer implementation."""

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