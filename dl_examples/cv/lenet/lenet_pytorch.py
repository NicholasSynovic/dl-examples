"""
Model Name: LeNet
Model Author(s): Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner,
Model Source Paper Title: Gradient-based learning applied to document recognition
Model Source Paper DOI: doi: 10.1109/5.726791

Model Implementation Author: Nicholas M. Synovic
"""

from pathlib import Path

import torch
from progress.bar import Bar
from torch import Tensor, nn
from torch.nn import Conv2d, CrossEntropyLoss, Dropout, Linear, MaxPool2d
from torch.nn import functional as F
from torch.optim import Adam
from torch.types import Number
from torch.utils.data import DataLoader

from dl_examples.cv.lenet.common import Common
from dl_examples.utils.metrics import Metrics

seed: int = 42
torch.manual_seed(seed=seed)


class LeNet(nn.Module):
    def __init__(
        self,
        tensorboardPath: Path,
        trainingDataLoader: DataLoader,
        testingDataLoader: DataLoader,
        validationDataLoader: DataLoader,
        outputPath: Path,
    ) -> None:
        super(LeNet, self).__init__()

        self.common: Common = Common()
        self.metrics: Metrics = Metrics(logDir=tensorboardPath)

        self.conv1: Conv2d = Conv2d(
            in_channels=1,
            out_channels=self.common.conv1_features,
            kernel_size=self.common.conv_kernelShape,
            stride=self.common.conv_strideShape,
            padding=self.common.padding.lower(),
        )
        self.maxPool1: MaxPool2d = MaxPool2d(
            kernel_size=self.common.maxPool_windowShape,
            stride=self.common.maxPool_strideShape,
        )
        self.conv2: Conv2d = Conv2d(
            in_channels=self.common.conv1_features,
            out_channels=self.common.conv2_features,
            kernel_size=self.common.conv_kernelShape,
            stride=self.common.conv_strideShape,
            padding=self.common.padding.lower(),
        )
        self.maxPool2: MaxPool2d = MaxPool2d(
            kernel_size=self.common.maxPool_windowShape,
            stride=self.common.maxPool_strideShape,
        )
        self.dense1: Linear = Linear(
            in_features=16 * 4 * 4,
            out_features=self.common.dense1_features,
        )
        self.dense2: Linear = Linear(
            in_features=self.common.dense1_features,
            out_features=self.common.dense2_features,
        )
        self.dense3: Linear = Linear(
            in_features=self.common.dense2_features,
            out_features=self.common.dense3_features,
        )

        self.dropout: Dropout = Dropout()

        self.lossFunction: CrossEntropyLoss = CrossEntropyLoss()

        self.trainingDataLoader: DataLoader = trainingDataLoader
        self.testingDataLoader: DataLoader = testingDataLoader
        self.validationDataLoader: DataLoader = validationDataLoader

        self.outputPath: Path = outputPath

    def forward(self, x: Tensor) -> Tensor:
        def convBlock(conv: Conv2d, maxPool: MaxPool2d, data: Tensor) -> Tensor:
            data = conv(data)
            data = F.relu(input=data)
            data = maxPool(data)
            return data

        def denseBlock(dense: Linear, data: Tensor) -> Tensor:
            data = dense(data)
            data = self.dropout(data)
            data = F.relu(input=data)
            return data

        x = convBlock(conv=self.conv1, maxPool=self.maxPool1, data=x)
        x = convBlock(conv=self.conv2, maxPool=self.maxPool2, data=x)
        x = torch.flatten(input=x, start_dim=1)
        x = denseBlock(dense=self.dense1, data=x)
        x = denseBlock(dense=self.dense2, data=x)
        return self.dense3(x)

    def predict(self, x: Tensor, y: Tensor) -> tuple[Tensor, int]:
        yPrediction: Tensor = self(x)

        loss: Tensor = self.lossFunction(yPrediction, y)
        correct: int = (
            (yPrediction.argmax(1) == y).type(torch.float).sum().item().__int__()
        )

        return (loss, correct)

    def run(
        self,
        epochs: int = 10,
        saveBestTrainingLoss: bool = True,
        saveBestTrainingAccuracy: bool = True,
    ) -> None:
        previousBestTrainingAccuracy: float = 0.0
        previousBestTrainingLoss: Number = 1
        trainingLoss: Number = -1

        optimizer: Adam = Adam(params=self.parameters(), lr=1e-3)

        def _trainingLoop(epoch: int) -> tuple[Number, float]:
            totalCorrectlyPredicted: int = 0
            loss: Tensor = Tensor()

            datasetSize: int = len(self.trainingDataLoader.dataset)

            for _, (x, y) in enumerate(self.trainingDataLoader):
                correct: int

                loss, correct = self.predict(x=x, y=y)

                totalCorrectlyPredicted += correct

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epochAccuracy: float = totalCorrectlyPredicted / datasetSize
            lossValue: Number = loss.item()

            self.metrics.logLoss(loss=lossValue, epoch=epoch)
            self.metrics.logEpochAccuracy(epochAccuracy=epochAccuracy, epoch=epoch)

            return (lossValue, epochAccuracy)

        def _validationLoop(epoch: int) -> None:
            totalCorrectlyPredicted: int = 0
            loss: Tensor = Tensor()

            datasetSize: int = len(self.validationDataLoader.dataset)

            for _, (x, y) in enumerate(self.validationDataLoader):
                correct: int

                loss, correct = self.predict(x=x, y=y)

                totalCorrectlyPredicted += correct

            epochAccuracy: float = totalCorrectlyPredicted / datasetSize

            self.metrics.logLoss(
                loss=loss.item(),
                epoch=epoch,
                mode="validation",
            )
            self.metrics.logEpochAccuracy(
                epochAccuracy=epochAccuracy,
                epoch=epoch,
                mode="validation",
            )

        def _conditionalSaveModelCheckpoint(
            condition1: float | Number, condition2: float | Number, path: Path
        ) -> bool:
            """Return True if condition1 > condition2, else false"""
            if condition1 > condition2:
                self.eval()
                torch.save(obj=self.state_dict(), f=path)
                self.train()
                return True
            return False

        with Bar(
            f"Training LeNet (PyTorch) for {epochs} epoch(s)... ", max=epochs
        ) as bar:
            epoch: int
            for epoch in range(epochs):
                epochAccuracy: float = 0.0

                trainingLoss, epochAccuracy = _trainingLoop(epoch=epoch)
                _validationLoop(epoch=epoch)

                if saveBestTrainingAccuracy:
                    checkpointPath: Path = Path(
                        self.outputPath, "lenet_bestAccuracy_epoch{epoch}.pth"
                    )
                    _conditionalSaveModelCheckpoint(
                        condition1=epochAccuracy,
                        condition2=previousBestTrainingAccuracy,
                        path=checkpointPath,
                    )

                if saveBestTrainingLoss:
                    checkpointPath: Path = Path(
                        self.outputPath, "lenet_bestLoss_epoch{epoch}.pth"
                    )
                    _conditionalSaveModelCheckpoint(
                        condition1=previousBestTrainingLoss,
                        condition2=trainingLoss,
                        path=checkpointPath,
                    )

                bar.next()
