from pathlib import Path
from typing import List

from torch import Generator
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


class MNIST(datasets.MNIST):
    def __init__(
        self,
        directory: Path,
        train: bool = True,
        batchSize: int = 64,
        shuffleDataBetweenEpochs: bool = True,
    ) -> None:
        self.batchSize: int = batchSize
        self.shuffleDataBetweenEpochs: bool = shuffleDataBetweenEpochs

        transformations: Compose = Compose(
            transforms=[
                ToTensor(),
                Normalize(mean=0.5, std=0.5),
            ]
        )

        super(MNIST, self).__init__(
            root=directory.__str__(), train=train, transform=transformations
        )

        self.dataloader: DataLoader = DataLoader(
            dataset=self,
            batch_size=self.batchSize,
            shuffle=self.shuffleDataBetweenEpochs,
        )

    def createTrainingValidationSplit(
        self, validationSizeRatio: float = 0.1
    ) -> tuple[DataLoader, DataLoader]:
        trainingSubset: Subset
        validationSubset: Subset

        datasetSize: int = len(self.dataloader.dataset)
        generator: Generator = Generator().manual_seed(42)

        validationSize: int = int(datasetSize * validationSizeRatio)
        sizes: List[int] = [datasetSize - validationSize, validationSize]

        sizes: List[int] = [datasetSize - validationSize, validationSize]
        trainingSubset, validationSubset = random_split(
            dataset=self,
            lengths=sizes,
            generator=generator,
        )

        trainingDataLoader: DataLoader = DataLoader(
            dataset=trainingSubset,
            batch_size=self.batchSize,
            shuffle=self.shuffleDataBetweenEpochs,
        )
        validationDataLoader: DataLoader = DataLoader(
            dataset=validationSubset,
            batch_size=self.batchSize,
            shuffle=self.shuffleDataBetweenEpochs,
        )

        return (trainingDataLoader, validationDataLoader)
