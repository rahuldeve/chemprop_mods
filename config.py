from dataclasses import dataclass
from enum import StrEnum, auto


class SplitType(StrEnum):
    RANDOM = auto()
    SCAFFOLD = auto()
    BUTINA = auto()


class ModelName(StrEnum):
    CHEMPROP = auto()
    CHEMPROP_MODDED = auto()


class Endpoint(StrEnum):
    """The six ADME endpoints, plus the in-vivo PK set.

    For the ADME members each value is the exact column header the endpoint is
    stored under in `datasets/ADME_public_set_3521.csv`; the member name is
    what you pass on the command line. Counts of non-null measurements differ
    wildly between them -- the two protein-binding endpoints have under 200
    compounds each, the rest have 2000-3000.

    `PK_AUC` is the odd one out: it comes from `datasets/PK.csv`, and its value
    is a display label rather than a column header, because the column there is
    raw `AUC` while what gets modelled is its log (see `load_data`).
    """

    HLM_CLINT = "LOG HLM_CLint (mL/min/kg)"
    RLM_CLINT = "LOG RLM_CLint (mL/min/kg)"
    MDR1_MDCK_ER = "LOG MDR1-MDCK ER (B-A/A-B)"
    SOLUBILITY = "LOG SOLUBILITY PH 6.8 (ug/mL)"
    PPB_HUMAN = "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)"
    PPB_RAT = "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)"
    PK_AUC = "LOG AUC"


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 40
    early_stopping_patience: int = 10
    random_state: int = 42
    weight_decay: float = 0.01
    mp_depth: int = 3
    ffn_zero_init: bool = True
    mp_dropout = 0.1


@dataclass
class EvalConfig:
    """Settings for the k-fold cross-validation loop itself."""

    n_folds: int = 10
    butina_cutoff: float = 0.65
    results_dir: str = "results"
    use_wandb: bool = False
