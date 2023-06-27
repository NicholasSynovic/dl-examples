class Common:
    def __init__(self, outputFeatueCount: int = 1000) -> None:
        self.kernelStride: tuple[int, int] = (3, 3)
        self.denseUnits: int = 4096
        self.outputFeatueCount = outputFeatueCount
