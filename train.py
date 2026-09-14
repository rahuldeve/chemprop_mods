from typing import Any, Iterator, Mapping, Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from config import SplitType, TrainConfig


def get_group_splitters_for_classification(random_state, n_outer):
    outer_splitter = StratifiedGroupKFold(
        n_splits=n_outer,
        shuffle=True,  # type: ignore
        random_state=random_state,  # type: ignore
    )
    # StratifiedGroupShuffleSplit does not exist, so use a 2-fold StratifiedGroupKFold for
    # the 50/50 val/test split (we take only the first fold). This keeps groups disjoint
    # across val and test too; plain StratifiedKFold ignores the `groups` arg and would
    # leak clusters between them. Equivalent to StratifiedGroupShuffleSplit with
    # test_size=0.5. ref: https://stackoverflow.com/a/79565815
    inner_spliter = StratifiedGroupKFold(
        n_splits=2, shuffle=True, random_state=random_state
    )
    return outer_splitter, inner_spliter


def get_random_splitters_for_classification(random_state, n_outer):
    outer_splitter = StratifiedKFold(
        n_splits=n_outer, shuffle=True, random_state=random_state
    )
    inner_spliter = StratifiedShuffleSplit(1, test_size=0.5, random_state=random_state)
    return outer_splitter, inner_spliter


def get_group_splitters_for_regression(random_state, n_outer):
    outer_splitter = GroupKFold(
        n_splits=n_outer,
        shuffle=True,  # type: ignore
        random_state=random_state,  # type: ignore
    )
    # GroupShuffleSplit would also work here, but a 2-fold GroupKFold gives the same
    # 50/50 val/test split (we take only the first fold) and mirrors the classification
    # splitter above. This keeps groups disjoint across val and test too; plain KFold
    # ignores the `groups` arg and would leak clusters between them. Stratification is
    # not available on a continuous target.
    inner_spliter = GroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    return outer_splitter, inner_spliter


def get_random_splitters_for_regression(random_state, n_outer):
    outer_splitter = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    inner_spliter = ShuffleSplit(1, test_size=0.5, random_state=random_state)
    return outer_splitter, inner_spliter


def generate_repeated_5x5_splits(
    df: pd.DataFrame, split_type: SplitType, random_state: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    rng = np.random.RandomState(random_state)
    for _ in range(5):
        randint = rng.randint(low=0, high=32767)

        match split_type:
            case SplitType.RANDOM:
                outer_splitter, inner_spliter = get_random_splitters_for_regression(
                    randint, n_outer=5
                )
                group_col_getter = lambda _df: None  # noqa: E731
                outer_split_iter = outer_splitter.split(
                    df, y=df["target"], groups=group_col_getter(df)
                )

            case SplitType.BUTINA:
                outer_splitter, inner_spliter = get_group_splitters_for_regression(
                    randint, n_outer=5
                )
                group_col_getter = lambda _df: _df["butina_cluster"]  # noqa: E731
                outer_split_iter = outer_splitter.split(
                    df, y=df["target"], groups=group_col_getter(df)
                )

            case _:
                raise ValueError(split_type)

        for train_idxs, val_test_idxs in outer_split_iter:
            train_df: pd.DataFrame = df.loc[train_idxs].reset_index(drop=True)  # type: ignore
            val_test_df = df.loc[val_test_idxs].reset_index(drop=True)

            val_idxs, test_idxs = next(
                inner_spliter.split(
                    val_test_df,
                    y=val_test_df["target"],
                    groups=group_col_getter(val_test_df),
                )
            )

            val_df = val_test_df.loc[val_idxs].reset_index(drop=True)
            test_df = val_test_df.loc[test_idxs].reset_index(drop=True)

            yield train_df, val_df, test_df


class TrainEvalFunc(Protocol):
    def __call__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_config: TrainConfig,
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...



def get_kfold_splitters(split_type: SplitType, n_folds: int, random_state: int):
    """Outer and inner splitters for `split_type`, plus the group column they need.

    Both levels come from the same splitter family on purpose. Carving a
    *random* val fold out of the training portion behind a Butina test fold
    would make early stopping an easier problem than scoring, and the two split
    types would stop being comparable -- the whole point of running both is
    that the only thing differing is how compounds are separated.
    """
    match split_type:
        case SplitType.BUTINA:
            outer = GroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)  # type: ignore[call-arg]
            inner = GroupKFold(n_splits=n_folds - 1, shuffle=True, random_state=random_state)  # type: ignore[call-arg]
            return outer, inner, lambda df: df["butina_cluster"]
        case SplitType.RANDOM:
            outer = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            inner = KFold(n_splits=n_folds - 1, shuffle=True, random_state=random_state)
            return outer, inner, lambda _df: None
        case _:
            raise ValueError(f"{split_type} is not implemented for the k-fold protocol")


def generate_kfold_splits(
    df: pd.DataFrame, split_type: SplitType, n_folds: int, random_state: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield `n_folds` (train, val, test) splits under `split_type`.

    The outer split makes each compound -- or, under Butina, each cluster --
    the test set exactly once, so the `n_folds` test scores partition the
    dataset and their mean is an estimate over every compound rather than over
    a resample.

    Validation is carved out of the *training* portion, not out of the test
    fold. Halving each held-out fold into val/test the way the 5x5 protocol
    does would leave only `1 / (2 * n_folds)` of the data to score on -- at 10
    folds that is 5%, or 8 compounds for the protein-binding endpoints. Instead
    the training portion is split `n_folds - 1` ways and one piece becomes val,
    which puts val and test at the same size and leaves roughly 80/10/10.
    """
    if n_folds < 3:
        raise ValueError(f"n_folds must be at least 3 to leave a val fold, got {n_folds}")

    outer, inner, get_groups = get_kfold_splitters(split_type, n_folds, random_state)

    for fit_idxs, test_idxs in outer.split(df, groups=get_groups(df)):
        fit_df = df.loc[fit_idxs].reset_index(drop=True)
        test_df = df.loc[test_idxs].reset_index(drop=True)

        train_idxs, val_idxs = next(inner.split(fit_df, groups=get_groups(fit_df)))
        train_df = fit_df.loc[train_idxs].reset_index(drop=True)
        val_df = fit_df.loc[val_idxs].reset_index(drop=True)

        yield train_df, val_df, test_df

def train_and_evaluate(
    df: pd.DataFrame,
    split_type: SplitType,
    model_train_eval_func: TrainEvalFunc,
    train_config: TrainConfig,
):
    splits = generate_kfold_splits(
        df, split_type, 10, random_state=train_config.random_state
    )
    
    for idx, (train_df, val_df, test_df) in enumerate(splits):
        if idx < 2:
            continue
        # print(val_df['target'].mean(), val_df['target'].std(), val_df['target'].min(), val_df['target'].max())
        # print(test_df['target'].mean(), test_df['target'].std(), test_df['target'].min(), test_df['target'].max())
        results_dict = model_train_eval_func(train_df, val_df, test_df, train_config)
        yield {"idx": idx} | dict(results_dict)
        break
