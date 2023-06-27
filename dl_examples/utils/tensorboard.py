from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter


class TensorBoard:
    def __init__(self, logDir: Path) -> None:
        self.writer: SummaryWriter = SummaryWriter(log_dir=logDir.__str__())

    def logScalar(
        self,
        tag: str,
        scalarValue: float,
        epoch: int,
    ) -> None:
        self.writer.add_scalar(
            tag=tag,
            scalar_value=scalarValue,
            global_step=epoch,
        )
        self.writer.flush()
