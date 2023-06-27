from pathlib import Path

from torchmetrics.classification import Accuracy, F1Score, Recall

from dl_examples.utils.tensorboard import TensorBoard


class Metrics(TensorBoard):
    def __init__(self, logDir: Path) -> None:
        super(Metrics, self).__init__(logDir=logDir)

    def logLoss(self, loss: float, epoch: int, mode: str = "train") -> None:
        self.logScalar(tag=f"Loss/{mode}", scalarValue=loss, epoch=epoch)

    def logEpochAccuracy(
        self, epochAccuracy: float, epoch: int, mode: str = "train"
    ) -> None:
        self.logScalar(
            tag=f"Epoch Accuracy/{mode}",
            scalarValue=epochAccuracy,
            epoch=epoch,
        )
