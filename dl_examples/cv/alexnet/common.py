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
