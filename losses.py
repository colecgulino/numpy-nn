"""Common loss functions."""

import numpy as np

import activations
import loss_base


class CrossEntropy(loss_base.Loss):
    """Implements cross entropy loss."""

    def __init__(
            self,
            reduction: loss_base.Reduction | str = loss_base.Reduction.MEAN,
            name: str = 'CrossEntropy',
            dim: int = 0
    ) -> None:
        super().__init__(reduction, name)
        self.dim = dim
        self.softmax = activations.Softmax(dim=self.dim)

    def forward(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Forward pass of the cross entropy loss.

        Implements:
            H(p, q) = -E_p[log(q)]
                    = - sum_x p(x) log(q(x))

        Implements:
            sum_i y[..., i] * log(softmax(logits)[..., i])

        Args:
            logits: Logits from the model of shape: [..., dim].
            y: True class labels of shape: [..., dim].

        Returns:
            Cross entropy loss of shape [...].
        """
        probs = self.softmax(logits)
        return -(y * np.log(probs)).sum(axis=self.dim)

    def backward_impl(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        probs = self.softmax(logits)
        return probs - y


class MSE(loss_base.Loss):
    """Implements mean squared error."""

    def __init__(
            self,
            reduction: loss_base.Reduction | str = loss_base.Reduction.MEAN,
            name: str = 'MSE',
            dim: int = 0
    ) -> None:
        super().__init__(reduction, name)
        self.dim = dim

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Forward pass of mean squared error.

        Implements: 1/I sum (y - x)^2

        Args:
            x: Output of the model of shape: [..., dim].
            y: Label to match to of shape: [..., dim].

        Returns:
            Mean squared error of shape [...].
        """
        if self.reduction == loss_base.Reduction.MEAN:
            return ((y - x)**2).mean(axis=self.dim)
        elif self.reduction == loss_base.Reduction.SUM:
            return ((y - x)**2).sum(axis=self.dim)
        else:
            raise ValueError(f'{self.reduction} not supported reduction.')

    def backward_impl(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Backwards implementation of the mean squared error.

        Derivation:
            dL/x = dL/dx 1/I sum (y - x)^2
                 = 1/I sum d[(y - x)^2]/dx
                 = 1/I sum -2(y - x)
                 = 1/I [-2I(y - x)]
                 = 2(x - y)
        """
        if self.reduction == loss_base.Reduction.MEAN:
            return 2 / x.shape[-1] * (x - y)
        elif self.reduction == loss_base.Reduction.SUM:
            return 2 * (x - y)
        else:
            raise ValueError(f'{self.reduction} not supported reduction.')
