"""Abstract Neural Network Layer class."""

import abc

import numpy as np

Cache = dict[str, np.ndarray]


def _shapes_equal(tensor1: np.ndarray, tensor2: np.ndarray) -> bool:
    """Helper function to check shapes of two tensors."""
    return bool(np.all(np.array(tensor1.shape) == np.array(tensor2.shape)))


def _update_parameters_for_layer(layer, parameters: dict[str, np.ndarray]) -> None:
    """Checks parameters and updates."""
    for k in layer._parameters.keys():
        if k not in parameters:
            raise ValueError(
                f'Parameter of network {k} not in {parameters.keys()}'
            )
        if not _shapes_equal(layer.parameters()[k], parameters[k]):
            raise ValueError(
                f'Parameter update of shape: {parameters[k].shape} does not match '
                f'original shape of {layer.parameters()[k].shape}.'
            )
        layer.parameters()[k] = parameters[k]


class Layer(abc.ABC):
    """The base layer for the neural network library. """

    def __init__(self, name: str = 'Layer') -> None:
        self.name = name
        self._parameters = {}

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)[0]

    def update_parameters(self, parameters: dict[str, np.ndarray]) -> None:
        """Update the parameters of the network given a dictionary of parameters."""
        # Update parameters on self.
        _update_parameters_for_layer(self, parameters)
        # Recurrsively call updates for all other nested Layers.
        layers = [v for v in self.__dict__.values() if isinstance(v, Layer)]
        for layer in layers:
            layer.update_parameters(parameters)

    def parameters(self) -> dict[str, np.ndarray]:
        """Returns the parameters of this network."""
        parameters = self._parameters
        # Find all the other modules that may have parameters.
        for v in self.__dict__.values():
            if isinstance(v, Layer):
                parameters.update(v.parameters())
        return parameters

    def set_parameter(self, parameter_name: str, parameters: np.ndarray) -> None:
        """Sets parameters into a parameter cache."""
        self._parameters[f'{self.name}/{parameter_name}'] = parameters

    def get_parameter(self, parameter_name: str) -> np.ndarray:
        """Returns the parameter suffix without the name prefix."""
        return self._parameters[f'{self.name}/{parameter_name}']

    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, Cache]:
        """Forward propogation of the network"""

    @abc.abstractmethod
    def backward(
        self,
        x: np.ndarray,
        cache: Cache,
        backwards_gradient: np.ndarray,
        gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards method of the layer.
        
        Args:
            x: Input to the layer.
            cache: Cache from the forward function to pass into the backward pass.
            backwards_gradient: Gradient from the previous layer that represents
                dout[prev_layer] / dout[this_layer].
            gradients: Mutable dictionary which should be updated with the
                gradients for all parameters in this layer.
            
        Returns:
            Gradients of this layer representing dout / din of the same shape
                as the input to be sent back to the next downstream layer.
        """
