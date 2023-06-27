# Citation:
# Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner,
# “Gradient-based learning applied to document recognition,”
# Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998,
# doi: 10.1109/5.726791.
#
# NOTE: The usage of ReLU and MaxPool for the activation function and
# subsampling was choosen based off of the Paper's With Code implementation
# (https://github.com/Elman295/Paper_with_code/blob/main/LeNet_5_Pytorch.ipynb)

from argparse import Namespace
from pathlib import Path

import torch
from progress.bar import Bar
from torch import Tensor, nn
from torch.nn import Conv2d, CrossEntropyLoss, Linear, MaxPool2d, functional
from torch.optim import Adam
from torch.types import Number
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dl_examples.args.lenet_args import getArgs
from dl_examples.datasetLoaders.mnist import MNIST

seed: int = 42
torch.manual_seed(seed=seed)


class Common:
    def __init__(self, outputFeatueCount: int = 10) -> None:
        self.padding: str = "VALID"

        self.maxPool_windowShape: tuple[int, int] = (2, 2)
        self.maxPool_strideShape: tuple[int, int] = (2, 2)

        self.conv_kernelShape: tuple[int, int] = (5, 5)
        self.conv_strideShape: tuple[int, int] = (1, 1)

        self.conv1_features: int = 6
        self.conv2_features: int = 16

        self.dense1_features: int = 120
        self.dense2_features: int = 84
        self.dense3_features: int = outputFeatueCount


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

        self.lossFunction: CrossEntropyLoss = CrossEntropyLoss()

        self.trainingDataLoader: DataLoader = trainingDataLoader
        self.testingDataLoader: DataLoader = testingDataLoader
        self.validationDataLoader: DataLoader = validationDataLoader

        self.outputPath: Path = outputPath
        self.writer: SummaryWriter = SummaryWriter(log_dir=tensorboardPath.__str__())

    def forward(self, x: Tensor) -> None:
        def convBlock(conv: Conv2d, maxPool: MaxPool2d, data: Tensor) -> Tensor:
            data = conv(data)
            data = functional.relu(input=data)
            data = maxPool(data)
            return data

        def denseBlock(dense: Linear, data: Tensor) -> Tensor:
            data = dense(data)
            data = functional.relu(input=data)
            return data

        x = convBlock(conv=self.conv1, maxPool=self.maxPool1, data=x)
        x = convBlock(conv=self.conv2, maxPool=self.maxPool2, data=x)
        x = torch.flatten(input=x, start_dim=1)
        x = denseBlock(dense=self.dense1, data=x)
        x = denseBlock(dense=self.dense2, data=x)
        return self.dense3(x)

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

        with Bar(
            f"Training LeNet (PyTorch) for {epochs} epoch(s)... ", max=epochs
        ) as bar:
            epoch: int
            for epoch in range(epochs):
                trainingCorrect: int = 0
                trainingAccuracy: float = 0.0
                validationAccuracy: float = 0.0
                validationLoss: Number = 0
                validationCorrect: int = 0

                for _, (x, y) in enumerate(self.trainingDataLoader):
                    yPrediction: Tensor = self(x)

                    loss: Tensor = self.lossFunction(yPrediction, y)
                    trainingLoss = loss.item()

                    trainingCorrect += (
                        (yPrediction.argmax(1) == y).type(torch.float).sum().item()
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                for _, (x, y) in enumerate(self.validationDataLoader):
                    with torch.no_grad():
                        yPrediction: Tensor = self(x)

                        validationLoss: Number = self.lossFunction(
                            yPrediction, y
                        ).item()
                        validationCorrect += (
                            (yPrediction.argmax(1) == y).type(torch.float).sum().item()
                        )

                trainingAccuracy = trainingCorrect / len(
                    self.trainingDataLoader.dataset
                )
                validationAccuracy = validationCorrect / len(
                    self.validationDataLoader.dataset
                )

                if saveBestTrainingAccuracy:
                    if trainingAccuracy > previousBestTrainingAccuracy:
                        self.eval()

                        previousBestTrainingAccuracy = trainingAccuracy
                        torch.save(
                            obj=self.state_dict(),
                            f=Path(
                                self.outputPath,
                                f"lenet_bestAccuracy_epoch{epoch}.pth",
                            ),
                        )

                        self.train()

                if saveBestTrainingLoss:
                    if trainingLoss < previousBestTrainingLoss:
                        self.eval()

                        previousBestTrainingLoss = trainingLoss
                        torch.save(
                            obj=self.state_dict(),
                            f=Path(
                                self.outputPath,
                                f"lenet_bestLoss_epoch{epoch}.pth",
                            ),
                        )

                        self.train()

                self.writer.add_scalar(
                    tag="Accuracy/train",
                    scalar_value=trainingAccuracy,
                    global_step=epoch,
                )
                self.writer.add_scalar(
                    tag="Loss/train",
                    scalar_value=trainingLoss,
                    global_step=epoch,
                )
                self.writer.add_scalar(
                    tag="Accuracy/validation",
                    scalar_value=validationAccuracy,
                    global_step=epoch,
                )
                self.writer.add_scalar(
                    tag="Loss/validation",
                    scalar_value=validationLoss,
                    global_step=epoch,
                )
                self.writer.flush()

                bar.next()


def main() -> None:
    args: Namespace = getArgs()

    trainingDataLoader: DataLoader
    validationDataLoader: DataLoader
    testingDataLoader: DataLoader

    datasetDirectory: Path = args.dataset[0]

    mnistTraining: MNIST = MNIST(
        directory=datasetDirectory, train=True, batchSize=args.batch_size[0]
    )
    mnistTesting: MNIST = MNIST(directory=datasetDirectory, train=False)

    splitDataLoader: tuple[
        DataLoader, DataLoader
    ] = mnistTraining.createTrainingValidationSplit()

    trainingDataLoader = splitDataLoader[0]
    validationDataLoader = splitDataLoader[1]
    testingDataLoader = mnistTesting.dataloader

    model: LeNet = LeNet(
        tensorboardPath=args.tensorboard[0],
        trainingDataLoader=trainingDataLoader,
        testingDataLoader=testingDataLoader,
        validationDataLoader=validationDataLoader,
        outputPath=args.output[0],
    )

    model.run()


if __name__ == "__main__":
    main()
