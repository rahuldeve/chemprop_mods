from collections.abc import Mapping
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
from chemprop.conf import DEFAULT_ATOM_FDIM
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import (
    MeanAggregation,
    NormAggregation,
    RegressionFFN,
    SumAggregation,
    UnscaleTransform,
    metrics,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from config import TrainConfig
from misc import seed_worker, set_seeds
from models.chemprop_modded_ref.mods import ModdedBondMessagePassing, ModdedMPNN
from models.rwse import compute_rwse, rwse_dims


def get_molecule_datapoint(row, train_config: TrainConfig):
    y = np.array([row["target"]])
    rwse = compute_rwse(row["mol"], train_config.rwse_k)
    if rwse is None:
        # Identical to the pre-RWSE call, so `rwse_k = 0` reproduces every
        # earlier run rather than passing a zero-width array through.
        return MoleculeDatapoint(mol=row["mol"], y=y)
    if train_config.rwse_at == "input":
        return MoleculeDatapoint(mol=row["mol"], y=y, V_f=rwse)
    return MoleculeDatapoint(mol=row["mol"], y=y, V_d=rwse)


def prepare_mol_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_config: TrainConfig,
):
    d_vf, _ = rwse_dims(train_config.rwse_k, train_config.rwse_at)

    def to_dp(row):
        return get_molecule_datapoint(row, train_config)

    train_df["mol_dp"] = train_df.apply(to_dp, axis=1)
    val_df["mol_dp"] = val_df.apply(to_dp, axis=1)
    test_df["mol_dp"] = test_df.apply(to_dp, axis=1)

    featurizer = SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=d_vf)

    train_mol_dataset = MoleculeDataset(train_df["mol_dp"], featurizer=featurizer)  # type: ignore
    scaler = train_mol_dataset.normalize_targets()

    val_mol_dataset = MoleculeDataset(val_df["mol_dp"], featurizer=featurizer)  # type: ignore
    val_mol_dataset.normalize_targets(scaler)

    test_mol_dataset = MoleculeDataset(test_df["mol_dp"], featurizer=featurizer)  # type: ignore

    train_mol_dataset.cache = True
    val_mol_dataset.cache = True

    return train_mol_dataset, val_mol_dataset, test_mol_dataset, scaler


def build_model(scaler, train_config: TrainConfig):
    d_vf, d_vd = rwse_dims(train_config.rwse_k, train_config.rwse_at)
    mp = ModdedBondMessagePassing(  # type: ignore
        d_v=DEFAULT_ATOM_FDIM + d_vf,
        d_vd=d_vd or None,
        depth=train_config.mp_depth,
        dropout=train_config.mp_dropout,
        zero_init=train_config.ffn_zero_init,
    )
    agg = NormAggregation()
    output_transform = UnscaleTransform.from_standard_scaler(scaler)
    # Sized from the encoder rather than left at the default 300: chemprop's
    # `W_d` is `Linear(d_h + d_vd, d_h + d_vd)`, so a readout-mode RWSE widens
    # the encoder output to `d_h + rwse_k`. At `rwse_k = 0` this is still 300,
    # so earlier runs reproduce.
    ffn = RegressionFFN(  # type: ignore
        n_tasks=1, input_dim=mp.output_dim, output_transform=output_transform
    )

    metric_list = [metrics.MAE(), metrics.RMSE(), metrics.R2Score()]
    return ModdedMPNN(
        mp,
        agg,
        ffn,
        metrics=metric_list,
        weight_decay=train_config.weight_decay,
    )


def train_and_evaluate_on_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_config: TrainConfig,
    **kwargs,
) -> Mapping[str, Any]:

    train_mol_ds, val_mol_ds, test_mol_ds, scaler = prepare_mol_datasets(
        train_df, val_df, test_df, train_config
    )

    set_seeds(train_config.random_state)

    train_loader = build_dataloader(
        train_mol_ds,
        batch_size=train_config.batch_size,
        num_workers=4,
        seed=train_config.random_state,
        worker_init_fn=seed_worker,
    )

    val_loader = build_dataloader(
        val_mol_ds, batch_size=train_config.batch_size, num_workers=4, shuffle=False
    )

    test_loader = build_dataloader(
        test_mol_ds, batch_size=train_config.batch_size, num_workers=4, shuffle=False
    )

    model = build_model(scaler, train_config)

    trainer = L.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=train_config.max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=True,
                patience=train_config.early_stopping_patience,
            ),
        ],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    results = trainer.test(model, dataloaders=test_loader, ckpt_path=None)[0]
    return results
