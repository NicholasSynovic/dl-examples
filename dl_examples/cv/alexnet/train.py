from argparse import Namespace
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from dl_examples.args.lenet_args import getArgs
from dl_examples.cv.lenet.lenet_pytorch import LeNet
from dl_examples.datasetLoaders.mnist import MNIST


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
