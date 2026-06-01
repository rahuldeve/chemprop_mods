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
    # Since StratifiedGroupShuffleSplit does not exist, we can use GroupShuffleSplit for
    # splitting val and test to get around this issue
    # ref: https://github.com/scikit-learn/scikit-learn/issues/12076#issuecomment-2047948563
    inner_spliter = StratifiedKFold(
        n_splits=int(1 / 0.5), shuffle=True, random_state=random_state
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
    # Since StratifiedGroupShuffleSplit does not exist, we can use GroupShuffleSplit for
    # splitting val and test to get around this issue
    # ref: https://github.com/scikit-learn/scikit-learn/issues/12076#issuecomment-2047948563
    inner_spliter = KFold(
        n_splits=int(1 / 0.5), shuffle=True, random_state=random_state
    )
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
                outer_splitter, inner_spliter = get_random_splitters_for_regression(
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


def train_and_evaluate(
    df: pd.DataFrame,
    split_type: SplitType,
    model_train_eval_func: TrainEvalFunc,
    train_config: TrainConfig,
):
    splits = generate_repeated_5x5_splits(
        df, split_type, random_state=train_config.random_state
    )
    
    for idx, (train_df, val_df, test_df) in enumerate(splits):
        results_dict = model_train_eval_func(train_df, val_df, test_df, train_config)
        yield dict(idx=idx) | dict(results_dict)
