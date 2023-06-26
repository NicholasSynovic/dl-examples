from pathlib import Path
from typing import List

from torchvision import datasets
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor


class MNIST(datasets.MNIST):
    def __init__(self, directory: Path, train: bool = True) -> None:
        transformations: Compose = Compose(
            transforms=[
                ToTensor(),
                Normalize(mean=0.5, std=0.5),
                Lambda(lambda x: x < 0.5),
            ]
        )

        super(MNIST, self).__init__(
            root=directory.__str__(), train=train, transform=transformations
        )
