from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Literal


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
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 10
    random_state: int = 42
    weight_decay: float = 1e-4
    mp_depth: int = 3
    ffn_zero_init: bool = True
    # Give each message passing step its own ResidualFFN instead of reusing one
    # block at every step. Costs (depth - 1) x ~270K parameters.
    ffn_per_level: bool = False
    # Give each message passing step its own `W_h` instead of sharing one.
    # Costs (depth - 1) x 90K parameters; sharing is what makes chemprop's
    # `mp_depth` free.
    w_h_per_level: bool = False
    # Number of random walk lengths encoded per atom. 0 disables RWSE entirely
    # and reproduces the plain featurizer exactly.
    rwse_k: int = 0
    # Where the encoding enters. "input" sizes the message passing matrices to
    # see it, so it conditions the chemistry; "readout" keeps message passing
    # purely chemical and adds it to the finalized atom representations, which
    # only does per-atom work under a non-linear aggregation.
    rwse_at: Literal["input", "readout"] = "input"
    # Readout aggregation. None keeps each model's own default -- mean for
    # stock chemprop, norm for the fork -- so every earlier result reproduces.
    # Setting it explicitly is what makes the two comparable: `norm` divides by
    # a fixed constant rather than the atom count, so it leaks molecule size
    # into the embedding magnitude, while `mean` does not.
    aggregation: Literal["mean", "sum", "norm"] | None = None
    # Divisor for `norm` aggregation. chemprop's default of 100 is ~4.3x the
    # mean heavy-atom count, so switching mean -> norm changes two things at
    # once: the readout stops being size-invariant, and it shrinks ~4.3x.
    # Nothing renormalises before the FFN (`MPNN.bn` is Identity by default),
    # so setting this near the mean atom count isolates the first effect.
    agg_norm: float = 100.0
    mp_dropout = 0.1


@dataclass
class EvalConfig:
    """Settings for the k-fold cross-validation loop itself."""

    n_folds: int = 10
    split_type: SplitType = SplitType.BUTINA
    """How compounds are separated across folds. `butina` holds out whole
    clusters, so the test set is chemistry the model has not seen; `random`
    scatters near-neighbours across train and test, which is the easier,
    in-distribution question. Defaults to `butina` so every earlier run and the
    commands in ABLATION_REPORT.md reproduce unchanged."""
    butina_cutoff: float = 0.65
    results_dir: str = "results"
    use_wandb: bool = False
