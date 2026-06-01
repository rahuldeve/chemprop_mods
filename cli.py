import tyro

from config import SplitType, TrainConfig


def load_data():
    import pandas as pd
    import rdkit.Chem as Chem

    from preprocessing import get_butina_clusters, mol_to_inchi, standardize

    df = pd.read_csv("./datasets/ADME_public_set_3521.csv")
    df = df.loc[:, ["SMILES", "LOG HLM_CLint (mL/min/kg)"]]
    df.columns = ["smiles", "target"]
    df = df.dropna(subset="target").reset_index(drop=True)

    df["mol"] = df["smiles"].map(standardize)
    df["inchi"] = df["mol"].map(mol_to_inchi)
    df["mol"] = df["inchi"].map(Chem.MolFromInchi)
    df["butina_cluster"] = get_butina_clusters(df["mol"])
    return df


def setup_wandb():
    import wandb

    wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
    run = wandb.init(project="chemprop_mods")
    run.mark_preempting()


def chemprop(split_type: SplitType, train_config: TrainConfig):
    import wandb

    from models import chemprop_ref as cp_ref
    from train import train_and_evaluate

    setup_wandb()

    df = load_data()
    results_iter = train_and_evaluate(
        df, split_type, cp_ref.train_and_evaluate_on_split, train_config
    )

    for result_dict in results_iter:
        wandb.log(result_dict | dict(model="chemprop"))
        print(result_dict)


def chemprop_modded(split_type: SplitType, train_config: TrainConfig):
    import wandb

    from models import chemprop_modded_ref as cpm_ref
    from train import train_and_evaluate

    setup_wandb()

    df = load_data()
    results_iter = train_and_evaluate(
        df, split_type, cpm_ref.train_and_evaluate_on_split, train_config
    )

    for result_dict in results_iter:
        wandb.log(result_dict | dict(model="chemprop-modded"))
        print(result_dict)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        dict(
            chemprop=chemprop,
            chemprop_modded=chemprop_modded,
        )
    )
