"""Functions to initialize parameters."""

import math

import numpy as np


def xavier_uniform(fan_in: int, fan_out: int, shape: list[int]) -> np.ndarray:
    """Implements Xavier uniform initialization for parameters.
    
    params ~ Uniform(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out)))

    Args:
        fan_in: Input dimension of the network.
        fan_out: Output dimension of the network.
    
    Returns:
        Xavier initialized network.
    """
    range = math.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-range, range, size=shape)
