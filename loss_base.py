"""Abstract module for neural network loss functions."""

import abc
import enum
import functools

import numpy as np


class Reduction(enum.Enum):
    """Reduction type enumaration."""
    MEAN = 'mean'
    SUM = 'sum'


class Loss(abc.ABC):
    """Abstract base class for loss functions."""

    def __init__(
            self,
            reduction: Reduction | str = Reduction.MEAN,
            name: str = 'Loss'
    ) -> None:
        self.name = name
        self.reduction = Reduction(reduction)

    def __call__(self, *args, **kwargs):
        loss = self.forward(*args, **kwargs)
        if self.reduction == Reduction.SUM:
            return loss.sum()
        if self.reduction == Reduction.MEAN:
            return loss.mean()
        raise ValueError(f'{self.reduction} not supported.')

    def backward(self, *args, **kwargs):
        """Backward outer function which also handles reduction"""
        backward_grad = self.backward_impl(*args, **kwargs)
        if self.reduction == Reduction.SUM:
            return backward_grad
        if self.reduction == Reduction.MEAN:
            b = functools.reduce(
                lambda x, y: x * y, backward_grad.shape[:-1], 1
            )
            return (1. / float(b)) * backward_grad
        raise ValueError(f'{self.reduction} not supported.')

    @abc.abstractmethod
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Forward function of the loss.
        
        Args:
            x: Output of the model of shape [..., dim].
            y: Target of the loss function of shape [..., dim].
        
        Returns:
            Value of the loss of shape [...]. 
        """

    @abc.abstractmethod
    def backward_impl(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Backward function of the loss.
        
        Args:
            x: Output of the model of shape [..., dim].
            y: Target of the loss function of shape [..., dim].
        
        Returns:
            Backward gradient representing dL / dx of shape [..., dim].
        """
