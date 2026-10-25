"""Configuration shared by the UI, examples, and evaluation service."""
from dataclasses import dataclass

@dataclass(frozen=True)
class EvaluationConfig:
    target: str
    prediction: str
    sensitive: tuple[str, ...]
    score: str | None = None
    min_group_size: int = 30
    medium_gap: float = 0.10
    high_gap: float = 0.20
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        if not self.sensitive or len(set(self.sensitive)) != len(self.sensitive):
            raise ValueError("Select unique sensitive attributes.")
        columns = [self.target, self.prediction, *self.sensitive]
        if self.score:
            columns.append(self.score)
        if len(set(columns)) != len(columns):
            raise ValueError("Target, prediction, score, and sensitive columns must be distinct.")
        if self.min_group_size < 1 or not 0 < self.medium_gap < self.high_gap <= 1:
            raise ValueError("Invalid group-size or risk-gap settings.")
        if not 2 <= self.calibration_bins <= 50:
            raise ValueError("Calibration bins must be between 2 and 50.")
