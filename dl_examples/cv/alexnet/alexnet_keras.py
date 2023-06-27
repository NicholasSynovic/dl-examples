"""
Model Name: AlexNet
Model Author(s): Krizhevsky et al.
Model Source Paper Title: ImageNet Classification with Deep Convolutional Neural Networks
Model Source Paper DOI: doi: 10.1145/3065386

Model Implementation Author: Nicholas M. Synovic
"""

from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, ReLU

from dl_examples.cv.alexnet.common import Common


class AlexNet:
    def __init__(self) -> None:
        self.common: Common = Common()

        self.model: Sequential = Sequential(name="AlexNet")

        self.model.add(layer=Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4)))
        self.model.add(layer=ReLU())
        self.model.add(layer=MaxPool2D())

        self.model.add(Conv2D(filters=256, kernel_size=(5, 5)))
        self.model.add(layer=ReLU())
        self.model.add(layer=MaxPool2D())

        self.model.add(
            layer=Conv2D(
                filters=384,
                kernel_size=self.common.kernelStride,
            )
        )
        self.model.add(layer=ReLU())

        self.model.add(
            layer=Conv2D(
                filters=384,
                kernel_size=self.common.kernelStride,
            )
        )
        self.model.add(layer=ReLU())

        self.model.add(
            layer=Conv2D(
                filters=256,
                kernel_size=self.common.kernelStride,
            )
        )
        self.model.add(layer=ReLU())
        self.model.add(layer=MaxPool2D())

        self.model.add(layer=Flatten())

        self.model.add(layer=Dense(units=self.common.denseUnits))
        self.model.add(layer=ReLU())

        self.model.add(layer=Dense(units=self.common.denseUnits))
        self.model.add(layer=ReLU())

        self.model.add(layer=Dropout(rate=0.5))

        self.model.add(
            layer=Dense(
                units=self.common.outputFeatueCount,
                activation="softmax",
            )
        )

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=True,
        )

    def run(self) -> None:
        pass
