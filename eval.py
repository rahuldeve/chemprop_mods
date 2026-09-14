# %%
import pandas as pd
import rdkit.Chem as Chem

from config import SplitType, TrainConfig
from misc import enable_determinism
from models import chemprop_modded_ref as cpm_ref
from models import chemprop_ref as cp_ref
from preprocessing import get_butina_clusters, mol_to_inchi, standardize
from train import generate_kfold_splits, train_and_evaluate

enable_determinism()


# %%
df = pd.read_csv("./datasets/ADME_public_set_3521.csv")
df = df.loc[:, ["SMILES", "LOG HLM_CLint (mL/min/kg)"]]
df.columns = ["smiles", "target"]
# df = df.sample(2000)
df = df.dropna(subset="target").reset_index(drop=True)

df["mol"] = df["smiles"].map(standardize)
df = df.dropna(subset='mol').reset_index(drop=True)
df["inchi"] = df["mol"].map(mol_to_inchi)
df["mol"] = df["inchi"].map(Chem.MolFromInchi)
df["butina_cluster"] = get_butina_clusters(df["mol"])
len(df)
# df

# %%
def evaluate(df, cp_config, cpm_config=None):
    if cpm_config is None:
            cpm_config = cp_config

    splits = generate_kfold_splits(
        df,
        split_type=SplitType.BUTINA,
        n_folds=3,
        random_state=cp_config.random_state,
    )

    results = []
    for fold, (train_df, val_df, test_df) in enumerate(splits):
        cp_results_dict = cp_ref.train_and_evaluate_on_split(train_df, val_df, test_df, cp_config)
        cp_results_dict['model'] = 'baseline' # type: ignore
        cp_results_dict['fold'] = fold # type: ignore
        results.append(cp_results_dict)

        cpm_results_dict = cpm_ref.train_and_evaluate_on_split(train_df, val_df, test_df, cpm_config)
        cpm_results_dict['model'] = 'modded' # type: ignore
        cpm_results_dict['fold'] = fold # type: ignore
        results.append(cpm_results_dict)

    return pd.DataFrame.from_records(results)

# %%
cfg = TrainConfig(max_epochs=20)
results_df = evaluate(df, cfg)

# %%
results_df.to_csv("./fin.csv")


