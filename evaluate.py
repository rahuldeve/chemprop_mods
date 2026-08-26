"""K-fold cross-validation over Butina-clustered splits.

`train.py` holds the repeated 5x5 nested protocol used for model development.
This module is the reporting protocol: a single pass of grouped k-fold in which
every compound is tested exactly once, which is what you want when comparing
endpoints or models on equal footing.
"""

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import Endpoint, EvalConfig, TrainConfig
from train import TrainEvalFunc


def generate_butina_kfold_splits(
    df: pd.DataFrame, n_folds: int, random_state: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield `n_folds` (train, val, test) splits with Butina clusters kept intact.

    The outer `GroupKFold` makes each cluster the test set exactly once, so the
    `n_folds` test scores partition the dataset and their mean is an estimate
    over every compound rather than over a resample.

    Validation is carved out of the *training* portion, not out of the test
    fold. Halving each held-out fold into val/test the way the 5x5 protocol
    does would leave only `1 / (2 * n_folds)` of the data to score on -- at 10
    folds that is 5%, or 8 compounds for the protein-binding endpoints. Instead
    the training portion is split `n_folds - 1` ways and one piece becomes val,
    which puts val and test at the same size and leaves roughly 80/10/10.
    """
    if n_folds < 3:
        raise ValueError(f"n_folds must be at least 3 to leave a val fold, got {n_folds}")

    outer = GroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)  # type: ignore[call-arg]
    inner = GroupKFold(n_splits=n_folds - 1, shuffle=True, random_state=random_state)  # type: ignore[call-arg]

    for fit_idxs, test_idxs in outer.split(df, groups=df["butina_cluster"]):
        fit_df = df.loc[fit_idxs].reset_index(drop=True)
        test_df = df.loc[test_idxs].reset_index(drop=True)

        train_idxs, val_idxs = next(
            inner.split(fit_df, groups=fit_df["butina_cluster"])
        )
        train_df = fit_df.loc[train_idxs].reset_index(drop=True)
        val_df = fit_df.loc[val_idxs].reset_index(drop=True)

        yield train_df, val_df, test_df


def cross_validate(
    df: pd.DataFrame,
    model_train_eval_func: TrainEvalFunc,
    train_config: TrainConfig,
    eval_config: EvalConfig,
) -> Iterator[Mapping[str, Any]]:
    """Fit and score one model per fold, yielding a result row as each finishes."""
    splits = generate_butina_kfold_splits(
        df, n_folds=eval_config.n_folds, random_state=train_config.random_state
    )

    for fold, (train_df, val_df, test_df) in enumerate(splits):
        print(
            f"\n=== fold {fold + 1}/{eval_config.n_folds} "
            f"(train={len(train_df)} val={len(val_df)} test={len(test_df)}, "
            f"{test_df['butina_cluster'].nunique()} test clusters) ==="
        )
        results = model_train_eval_func(train_df, val_df, test_df, train_config)

        yield {
            "fold": fold,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        } | dict(results)


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and standard error across folds for every metric column.

    The std is over folds, so it mixes real cluster-to-cluster difficulty with
    run-to-run training noise. Treat it as a spread, not as a confidence
    interval on the mean.
    """
    metric_cols = [c for c in results_df.columns if c.startswith("test/")]
    metrics = results_df[metric_cols]

    # Built in one shot rather than assembled then reindexed: pandas ships no
    # overloads for `DataFrame.__getitem__`, so any `df[...]` reads back as
    # `Series | DataFrame` and would make the return type unverifiable.
    mean, std, count = metrics.mean(), metrics.std(), metrics.count()
    return pd.DataFrame(
        {"mean": mean, "std": std, "sem": std / np.sqrt(count), "count": count}
    )


def format_summary(endpoint: Endpoint, model_name: str, stats: pd.DataFrame) -> str:
    lines = [
        "",
        f"{model_name} | {endpoint.name} | {int(stats['count'].iloc[0])}-fold Butina CV",
        f"  {endpoint.value}",
        "",
    ]
    for metric, row in stats.iterrows():
        name = str(metric).removeprefix("test/")
        lines.append(f"  {name:>6}  {row['mean']:8.4f} +/- {row['std']:.4f}  (sem {row['sem']:.4f})")
    return "\n".join(lines) + "\n"


def save_results(
    results_df: pd.DataFrame,
    stats: pd.DataFrame,
    endpoint: Endpoint,
    model_name: str,
    eval_config: EvalConfig,
    train_config: TrainConfig,
) -> Path:
    """Write per-fold scores and the run's settings to `results_dir`."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(eval_config.results_dir) / f"{stamp}_{model_name}_{endpoint.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(run_dir / "folds.csv", index=False)
    stats.to_csv(run_dir / "summary.csv")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "endpoint": endpoint.name,
                "target_column": endpoint.value,
                "eval_config": vars(eval_config),
                "train_config": vars(train_config) | {"mp_dropout": train_config.mp_dropout},
            },
            indent=2,
        )
    )
    return run_dir
