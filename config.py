from dataclasses import dataclass
from enum import StrEnum, auto


class SplitType(StrEnum):
    RANDOM = auto()
    SCAFFOLD = auto()
    BUTINA = auto()


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 40
    early_stopping_patience: int = 10
    random_state: int = 42
    weight_decay: float = 0.01
    mp_dropout = 0.1
