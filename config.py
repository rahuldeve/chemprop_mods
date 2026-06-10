from dataclasses import dataclass
from enum import StrEnum, auto


class SplitType(StrEnum):
    RANDOM = auto()
    SCAFFOLD = auto()
    BUTINA = auto()


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 10
    random_state: int = 42
