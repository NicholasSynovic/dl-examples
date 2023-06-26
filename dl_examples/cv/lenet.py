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
from flax import linen
from flax.linen import Conv, Dense, max_pool, relu
from jax import Array
from jax import numpy as jnp
from jax.random import KeyArray, PRNGKey
from torch import Tensor, nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

from dl_examples.args.lenet_args import getArgs

seed: int = 42
rngKey: KeyArray = PRNGKey(seed=seed)
torch.manual_seed(seed=seed)


class Common:
    def __init__(self, outputFeatueCount: int = 10) -> None:
        self.jaxPadding: str = "VALID"

        self.maxPool_windowShape: tuple[int, int] = (2, 2)
        self.maxPool_strideShape: tuple[int, int] = (2, 2)

        self.conv_kernelShape: tuple[int, int] = (5, 5)
        self.conv_strideShape: tuple[int, int] = (1, 1)

        self.conv1_features: int = 6
        self.conv2_features: int = 16

        self.dense1_features: int = 120
        self.dense2_features: int = 84
        self.dense3_features: int = outputFeatueCount


class LeNet_Jax(linen.Module):
    def __init__(self) -> None:
        super(LeNet_Jax, self).__init__()

        self.common: Common = Common()

    def setup(self) -> None:
        self.conv1: Conv = Conv(
            features=self.common.conv1_features,
            kernel_size=self.common.conv_kernelShape,
            strides=self.common.conv_strideShape,
            padding=self.common.jaxPadding,
        )
        self.conv2: Conv = Conv(
            features=self.common.conv2_features,
            kernel_size=self.common.conv_kernelShape,
            strides=self.common.conv_strideShape,
            padding=self.common.jaxPadding,
        )
        self.dense1: Dense = Dense(features=self.common.dense1_features)
        self.dense2: Dense = Dense(features=self.common.dense2_features)
        self.dense3: Dense = Dense(features=self.common.dense3_features)

    def __call__(self, x: Array) -> Array:
        def convBlock(conv: Conv, data: Array) -> Array:
            data = conv(data)
            data = relu(data)
            data = max_pool(
                data,
                window_shape=self.common.maxPool_windowShape,
                strides=self.common.maxPool_strideShape,
                padding=self.common.jaxPadding,
            )
            return data

        def denseBlock(dense: Dense, data: Array) -> Array:
            data = dense(data)
            data = relu(data)
            return data

        x = convBlock(conv=self.conv1, data=x)
        x = convBlock(conv=self.conv2, data=x)
        x = denseBlock(dense=self.dense1, data=x)
        x = denseBlock(dense=self.dense2, data=x)
        x = self.dense3(inputs=x)

        return x


class LeNet_PyTorch(nn.Module):
    def __init__(self) -> None:
        super(LeNet_PyTorch, self).__init__()

        self.common: Common = Common()


def getMNIST(rootDirectory: Path = Path(".")) -> None:
    MNIST(root=rootDirectory.__str__(), download=True)


def loadMNIST(
    tranformation: Compose,
    rootDirectory: Path = Path("."),
    train: bool = True,
) -> MNIST:
    mnist: MNIST = MNIST(
        root=rootDirectory.__str__(),
        train=train,
        download=False,
        transform=tranformation,
    )
    return mnist


def main() -> None:
    args: Namespace = getArgs()
    datasetDirectory: Path = args.dataset[0]

    batchSize: int = args.batch_size[0]
    inputShape: tuple[int, int, int, int] = (
        batchSize,
        28,
        28,
        1,
    )  # (batch_size, height, width, channels)

    binaryTransform: Compose = Compose(
        transforms=[ToTensor(), Lambda(lambda x: x > 0.5)]
    )

    model: LeNet = LeNet()

    getMNIST(rootDirectory=datasetDirectory)

    mnistTrain: MNIST = loadMNIST(
        tranformation=binaryTransform,
        rootDirectory=datasetDirectory,
    )
    mnistTest: MNIST = loadMNIST(
        tranformation=binaryTransform,
        rootDirectory=datasetDirectory,
        train=False,
    )

    trainingDataLoader: DataLoader = DataLoader(
        dataset=mnistTrain, batch_size=32, shuffle=True
    )

    data: Array = jnp.ones(inputShape)

    # params = model.init(rngs=rngKey, x=data)
    print(model.tabulate(rngs=rngKey, x=data))


if __name__ == "__main__":
    main()
