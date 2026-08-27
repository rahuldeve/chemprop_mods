import tyro

from config import Endpoint, EvalConfig, ModelName, SplitType, TrainConfig


def load_data(endpoint: Endpoint, butina_cutoff: float = 0.65):
    """Load the ADME set, keeping the compounds measured for `endpoint`.

    Butina clusters are assigned *after* dropping the unmeasured rows, so the
    groups describe the dataset actually being split. The endpoints do not
    share a compound set -- HLM has 3087 measurements and rat PPB has 168 --
    so clustering the full file once would leave most endpoints with groups
    that are mostly empty.
    """
    import numpy as np
    import pandas as pd
    import rdkit.Chem as Chem

    from preprocessing import get_butina_clusters, mol_to_inchi, standardize

    if endpoint is Endpoint.PK_AUC:
        # Separate export with its own conventions: SMILES live in `mol`, and
        # AUC is raw where every ADME column is already logged. It spans five
        # orders of magnitude (0 to 110845) with nine true zeros, so the +1
        # offset is what keeps the log defined -- skew 3.7 -> -0.4.
        df = pd.read_csv("./datasets/PK.csv").loc[:, ["mol", "AUC"]]
        df.columns = ["smiles", "target"]
        df = df.dropna(subset=["smiles", "target"]).reset_index(drop=True)
        df["target"] = np.log10(df["target"] + 1)
    else:
        df = pd.read_csv("./datasets/ADME_public_set_3521.csv")
        df = df.loc[:, ["SMILES", endpoint.value]]
        df.columns = ["smiles", "target"]
        df = df.dropna(subset="target").reset_index(drop=True)

    df["mol"] = df["smiles"].map(standardize)
    df = df.dropna(subset="mol").reset_index(drop=True)
    df["inchi"] = df["mol"].map(mol_to_inchi)
    df = df.dropna(subset="inchi").reset_index(drop=True)
    df["mol"] = df["inchi"].map(Chem.MolFromInchi)
    df = df.dropna(subset="mol").reset_index(drop=True)
    df["butina_cluster"] = get_butina_clusters(df["mol"], cutoff=butina_cutoff)
    return df


def get_train_eval_func(model: ModelName):
    """Resolve a model name to its train/eval entry point.

    Imported lazily -- pulling in torch, lightning and chemprop takes seconds,
    and `--help` should not pay for it.
    """
    match model:
        case ModelName.CHEMPROP:
            from models import chemprop_ref

            return chemprop_ref.train_and_evaluate_on_split
        case ModelName.CHEMPROP_MODDED:
            from models import chemprop_modded_ref

            return chemprop_modded_ref.train_and_evaluate_on_split
        case _:
            raise ValueError(model)


def setup_wandb():
    import wandb

    wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
    run = wandb.init(project="chemprop_mods")
    run.mark_preempting()


def evaluate(
    endpoint: Endpoint,
    model: ModelName,
    train_config: TrainConfig,
    eval_config: EvalConfig,
) -> None:
    """Cross-validate one model on one ADME endpoint over Butina-grouped folds."""
    # `endpoint` and `model` are deliberately required. A score is meaningless
    # without both, and a default would let a run be mislabelled in the results
    # dir or in wandb without anyone noticing. The two config groups carry no
    # default *object* either, but their fields keep the dataclass defaults --
    # so `--endpoint` and `--model` are the only mandatory flags.
    import pandas as pd

    from evaluate import cross_validate, format_summary, save_results, summarize

    df = load_data(endpoint, butina_cutoff=eval_config.butina_cutoff)
    print(
        f"{endpoint.name}: {len(df)} compounds, "
        f"{df['butina_cluster'].nunique()} Butina clusters "
        f"(cutoff {eval_config.butina_cutoff}), "
        f"target mean {df['target'].mean():.3f} std {df['target'].std():.3f}"
    )

    if eval_config.use_wandb:
        import wandb

        setup_wandb()
        wandb.config.update(
            {
                "model": str(model),
                "endpoint": endpoint.name,
                "n_folds": eval_config.n_folds,
            }
        )

    rows = []
    for row in cross_validate(df, get_train_eval_func(model), train_config, eval_config):
        print(row)
        rows.append(row)
        if eval_config.use_wandb:
            import wandb

            wandb.log(dict(row) | {"model": str(model), "endpoint": endpoint.name})

    results_df = pd.DataFrame(rows)
    stats = summarize(results_df)
    print(format_summary(endpoint, str(model), stats))

    run_dir = save_results(
        results_df, stats, endpoint, str(model), eval_config, train_config
    )
    print(f"wrote {run_dir}")

    if eval_config.use_wandb:
        import wandb

        wandb.log({f"cv/{m}": r["mean"] for m, r in stats.iterrows()})
        wandb.finish()


def chemprop(
    split_type: SplitType,
    train_config: TrainConfig,
    endpoint: Endpoint = Endpoint.PPB_RAT,
):
    import wandb

    from train import train_and_evaluate

    setup_wandb()

    df = load_data(endpoint)
    results_iter = train_and_evaluate(
        df, split_type, get_train_eval_func(ModelName.CHEMPROP), train_config
    )

    for result_dict in results_iter:
        wandb.log(result_dict | {"model": "chemprop"})
        print(result_dict)


def chemprop_modded(
    split_type: SplitType,
    train_config: TrainConfig,
    endpoint: Endpoint = Endpoint.PPB_RAT,
):
    import wandb

    from train import train_and_evaluate

    setup_wandb()

    df = load_data(endpoint)
    results_iter = train_and_evaluate(
        df, split_type, get_train_eval_func(ModelName.CHEMPROP_MODDED), train_config
    )

    for result_dict in results_iter:
        wandb.log(result_dict | {"model": "chemprop-modded"})
        print(result_dict)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "evaluate": evaluate,
            "chemprop": chemprop,
            "chemprop_modded": chemprop_modded,
        }
    )
