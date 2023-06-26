from pathlib import Path

import numpy as np
from jax import Array
from jax import numpy as jnp
from numpy import ndarray
from torch.utils.data import DataLoader
from torchvision import datasets


class MNIST(datasets.MNIST):
    def __init__(self, path: Path, train: bool, batchSize: int = 32) -> None:
        super(self.__class__, self).__init__(
            root=path.__str__(),
            train=train,
            download=True,
        )

        self.dataloader: DataLoader = DataLoader(
            dataset=self,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
        )
