"""Convolutional network layers."""

import itertools

import numpy as np

import initializers
import layer


def pad_image(image: np.ndarray, padding: int, padding_value: float = 0.) -> np.ndarray:
    """Pads an image equally on all sides."""
    batch, height, width, channels = image.shape
    row_padding = np.ones(shape=(batch, padding, width, channels)) * padding_value
    column_padding = np.ones(shape=(batch, height + 2 * padding, padding, channels)) * padding_value
    image = np.concatenate((row_padding, image, row_padding), axis=1)
    image = np.concatenate((column_padding, image, column_padding), axis=2)
    return image


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution 2D function.
    
    Args:
        image: Image input of shape [B, H, W, in_channels, 1].
        kernel: Filter to convolve over image of shape [1, 1, K, K, in_channels, out_channels].
    
    Returns:
        Output of the 2d convolution filter of shape
            [B, new_H, new_W, in_channels, out_channels].
    """
    k = kernel.shape[2]
    hs = range(image.shape[1] - k + 1)
    ws = range(image.shape[2] - k + 1)
    chunks = []
    for h, w in itertools.product(hs, ws):
        chunks.append(image[:, h:h + k, w:w + k])
    image = np.stack(chunks, axis=1)
    return image * kernel


def pad_input_sequence(seq: np.ndarray, padding: int, value: float = 0.) -> np.ndarray:
    """Pad input sequence."""
    batch, _, channels = seq.shape
    # Shape: [B, padding, C].
    pad = np.ones((batch, padding, channels), dtype=seq.dtype) * value
    # Shape: [B, T + 2 * padding, C].
    return np.concatenate((pad, seq, pad), axis=1)


def convolve_1d(seq: np.ndarray, kernel: np.ndarray, kernel_dim: int) -> np.ndarray:
    """Functional for the convolution operator."""
    time = seq.shape[1]
    k = kernel.shape[kernel_dim]
    # Shape: [B, num_chunks, K, in_channels].
    chunks = []
    for chunk_start in range(0, time - k + 1, 1):
        chunks.append(seq[:, chunk_start:chunk_start + k])
    seq = np.stack(chunks, axis=1)
    # Shape: [B, num_chunks, K, in_channels, out_channels].
    output = seq * kernel
    return output


class Conv2D(layer.Layer):
    """2D Convolutional network."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int = 1,
            padding_value: float = 0.,
            name: str = 'Conv2D'
    ) -> None:
        super().__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_value = padding_value
        self.stride = 1
        self.set_parameter(
            'W',
            initializers.xavier_uniform(
                in_channels,
                out_channels,
                shape=(kernel_size, kernel_size, in_channels, out_channels)
            )
        )
        self.set_parameter('b', np.zeros([self.out_channels]))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Forward propogation of the network"""
        b, h, w = x.shape[:3]
        x = pad_image(x, self.padding, self.padding_value)
        weight, bias = self.get_parameter('W'), self.get_parameter('b')
        output = convolve_2d(x[..., None], np.expand_dims(weight, axis=[0, 1]))
        output = output.sum(2).sum(2).sum(-2)
        new_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        new_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
        return np.reshape(output, [b, new_h, new_w, self.out_channels]) + bias, {}

    def backward(
        self,
        x: np.ndarray,
        cache: layer.Cache,
        backwards_gradient: np.ndarray,
        gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backwards pass of convolution 1d.
        
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
        del cache
        image = pad_image(x, self.padding, self.padding_value)

        gradients[f'{self.name}/b'] = backwards_gradient.sum(0).sum(0).sum(0)

        # Convolve dout with the padded input.
        dw = convolve_2d(image[..., None], np.expand_dims(backwards_gradient, axis=[1, -2]))
        dw = dw.sum(0).sum(1).sum(1)
        weight = self.get_parameter('W')
        gradients[f'{self.name}/W'] = dw.reshape(weight.shape)

        # Convolve the rotated filter by 180deg.
        rotated_filter = np.rot90(weight, k=2, axes=(0, 1))
        padding_needed = (self.kernel_size + x.shape[1] - 1 - backwards_gradient.shape[1]) // 2
        padded_dout = pad_image(backwards_gradient, padding=padding_needed)
        din = convolve_2d(
            np.expand_dims(padded_dout, axis=-2),
            np.expand_dims(rotated_filter, axis=[0, 1])
        )
        din = din.sum(2).sum(2).sum(-1).reshape(x.shape)
        return din


class Conv1D(layer.Layer):
    """Convolutional 1D layer."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            padding_value: float = 0.,
            name: str = 'Conv1D'
    ) -> None:
        super().__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.padding_value = padding_value
        self.kernel_size = kernel_size
        self.set_parameter(
            'W',
            initializers.xavier_uniform(
                self.in_channels,
                self.out_channels,
                shape=(kernel_size, in_channels, out_channels)
            )
        )
        self.set_parameter('b', np.zeros((out_channels)))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, layer.Cache]:
        """Forward pass for the Conv1D layer.
        
        Args:
            x: Input to the module of shape [B, T, in_channels].
        
        Returns:
            Output of the module of shape [B, new_T, out_channels].
        """
        weight, bias = self.get_parameter('W'), self.get_parameter('b')
        # Pad the input to new shape of [B, T + 2 * padding, in_channels].
        x = pad_input_sequence(x, padding=self.padding, value=self.padding_value)
        # Run the convolution operator over the sequence and the kernel.
        output = convolve_1d(x[..., None], np.expand_dims(weight, axis=[0, 1]), kernel_dim=2)
        output = output.sum(axis=2).sum(axis=2)
        return output + bias, {}

    def backward(
        self,
        x: np.ndarray,
        cache: layer.Cache,
        backwards_gradient: np.ndarray,
        gradients: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Backward pass for a 1D convolutional network."""
        del cache
        # Pad the input to new shape of [B, T + 2 * padding, in_channels].
        x = pad_input_sequence(x, padding=self.padding, value=self.padding_value)

        # Calculate the gradients of the bias.
        gradients[f'{self.name}/b'] = backwards_gradient.sum(0).sum(0)

        # Calculate the weights as a convolution of the error signal over the
        # input.
        dw = convolve_1d(x[..., None], np.expand_dims(
            backwards_gradient, axis=[1, -2]), kernel_dim=2
        )
        gradients[f'{self.name}/W'] = dw.sum(0).sum(1)

        din = np.zeros_like(x)
        padding_needed = (self.kernel_size + 1) // 2 - self.padding
        dout_padded = pad_input_sequence(
            backwards_gradient, padding=padding_needed, value=self.padding_value
        )

        flipped_weight = np.flip(self.get_parameter('W'), axis=0)

        kernel = np.expand_dims(flipped_weight, axis=[0, 1])
        din = convolve_1d(dout_padded[..., None, :], kernel, kernel_dim=2).sum(2).sum(-1)
        return din
