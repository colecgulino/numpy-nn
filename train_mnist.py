"""Simple explanatory mnist example."""

import torch
from torch.nn import functional as F
import torchvision

from nn import activations
from nn import dense
from nn import losses
from nn import optimizers


def get_dataset(batch_size: int = 64):
    """Gets the MNIST dataset."""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(
        'data/', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader


def sgd_test():
    """Tests optimization on MNIST."""
    batch_size = 64

    layers = [
        dense.Dense(28 * 28, 256),
        activations.ReLU(),
        dense.Dense(256, 10, name='Dense2')
    ]

    loss_fn = losses.CrossEntropy(reduction='mean', dim=-1)
    all_parameters = {}
    for layer in layers:
        all_parameters.update(layer.parameters())
    optimizer2 = optimizers.Adam(1e-3, all_parameters, loss_fn)
    for e in range(5):
        all_losses = []
        for i, (x, y) in enumerate(get_dataset(batch_size)):
            y = F.one_hot(y, num_classes=10).numpy()
            b = x.shape[0]
            x = x.contiguous().view(b, -1).numpy()
            output = optimizer2.step(x, y, layers)
            all_losses.append(output.loss)
        print(f'Epoch: {e} | Average loss: {sum(all_losses) / len(all_losses)}')


if __name__ == '__main__':
    sgd_test()
