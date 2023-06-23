# Citation:
# Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner,
# “Gradient-based learning applied to document recognition,”
# Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998,
# doi: 10.1109/5.726791.
#
# NOTE: The usage of ReLU and MaxPool for the activation function and
# subsampling was choosen based off of the Paper's With Code implementation
# (https://github.com/Elman295/Paper_with_code/blob/main/LeNet_5_Pytorch.ipynb)

from pathlib import Path

from flax import linen
from flax.linen import Conv, Dense, max_pool, relu
from jax import Array
from torchvision.datasets import MNIST


class LeNet(linen.Module):
    def setup(self) -> None:
        kernelSize: tuple[int, int] = (5, 5)
        strideSize: tuple[int, int] = (1, 1)
        paddingType: str = "VALID"

        self.conv1: Conv = Conv(
            features=6,
            kernel_size=kernelSize,
            strides=strideSize,
            padding=paddingType,
        )
        self.conv2: Conv = Conv(
            features=16,
            kernel_size=kernelSize,
            strides=strideSize,
            padding=paddingType,
        )
        self.dense1: Dense = Dense(features=120)
        self.dense2: Dense = Dense(features=84)
        self.dense3: Dense = Dense(features=10)

    def __call__(self, x: Array) -> Array:
        def convBlock(conv: Conv, data: Array) -> Array:
            data = conv(inputs=data)
            data = relu(inputs=data)
            data = max_pool(
                inputs=data,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="VALID",
            )
            return data

        def denseBlock(dense: Dense, data: Array) -> Array:
            data = dense(inputs=data)
            data = relu(data)
            return data

        x = convBlock(conv=self.conv1, data=x)
        x = convBlock(conv=self.conv2, data=x)
        x = denseBlock(dense=self.dense1, data=x)
        x = denseBlock(dense=self.dense2, data=x)
        x = self.dense3(inputs=x)

        return x


def getMNIST(rootDirectory: Path = Path(".")) -> None:
    MNIST(root=rootDirectory.__str__(), download=True)


def loadMNIST_train(rootDirectory: Path = Path(".")) -> MNIST:
    mnist: MNIST = MNIST(
        root=rootDirectory.__str__(),
        train=True,
        download=False,
    )
    return mnist


def main() -> None:
    model: LeNet = LeNet()

    getMNIST()
    mnistTrain: MNIST = loadMNIST_train()

    print(mnistTrain)


if __name__ == "__main__":
    main()
