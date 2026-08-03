from collections.abc import Mapping
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import (
    BondMessagePassing,
    MeanAggregation,
    NormAggregation,
    RegressionFFN,
    UnscaleTransform,
    metrics,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from config import TrainConfig
from misc import seed_worker, set_seeds


def get_molecule_datapoint(row):
    return MoleculeDatapoint(mol=row["mol"], y=np.array([row["target"]]))


def prepare_mol_datasets(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    train_df["mol_dp"] = train_df.apply(get_molecule_datapoint, axis=1)
    val_df["mol_dp"] = val_df.apply(get_molecule_datapoint, axis=1)
    test_df["mol_dp"] = test_df.apply(get_molecule_datapoint, axis=1)

    featurizer = SimpleMoleculeMolGraphFeaturizer()

    train_mol_dataset = MoleculeDataset(train_df["mol_dp"], featurizer=featurizer)  # type: ignore
    scaler = train_mol_dataset.normalize_targets()

    val_mol_dataset = MoleculeDataset(val_df["mol_dp"], featurizer=featurizer)  # type: ignore
    val_mol_dataset.normalize_targets(scaler)

    test_mol_dataset = MoleculeDataset(test_df["mol_dp"], featurizer=featurizer)  # type: ignore

    train_mol_dataset.cache = True
    val_mol_dataset.cache = True

    return train_mol_dataset, val_mol_dataset, test_mol_dataset, scaler


def build_model(scaler, train_config: TrainConfig):
    mp = BondMessagePassing(dropout=train_config.mp_dropout)  # type: ignore
    agg = MeanAggregation()
    output_transform = UnscaleTransform.from_standard_scaler(scaler)
    ffn = RegressionFFN(n_tasks=1, output_transform=output_transform)  # type: ignore

    metric_list = [metrics.MAE(), metrics.RMSE(), metrics.R2Score()]
    return MPNN(
        mp,
        agg,
        ffn,
        metrics=metric_list,
    )


def train_and_evaluate_on_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_config: TrainConfig,
    **kwargs: Any,
) -> Mapping[str, Any]:

    train_mol_ds, val_mol_ds, test_mol_ds, scaler = prepare_mol_datasets(
        train_df, val_df, test_df
    )

    set_seeds(train_config.random_state)

    train_loader = build_dataloader(
        train_mol_ds,
        batch_size=train_config.batch_size,
        num_workers=8,
        seed=train_config.random_state,
        worker_init_fn=seed_worker,
    )

    val_loader = build_dataloader(
        val_mol_ds, batch_size=train_config.batch_size, num_workers=8, shuffle=False
    )

    test_loader = build_dataloader(
        test_mol_ds, batch_size=train_config.batch_size, num_workers=8, shuffle=False
    )

    model = build_model(scaler, train_config)

    trainer = L.Trainer(
        logger=None,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=train_config.max_epochs,
        num_sanity_val_steps=0,
        # callbacks=[
        #     EarlyStopping(
        #         monitor="val_loss",
        #         mode="min",
        #         verbose=True,
        #         patience=train_config.early_stopping_patience,
        #     ),
        # ],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    results = trainer.test(model, dataloaders=test_loader, ckpt_path=None)[0]
    return results
