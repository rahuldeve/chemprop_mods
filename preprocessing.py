import numpy as np
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.ML.Cluster import Butina
from rdkit.rdBase import BlockLogs


def standardize(smiles):
    with BlockLogs():
        params = Chem.SmilesParserParams()  # type: ignore
        params.removeHs = False
        mol = Chem.MolFromSmiles(smiles, params)  # type: ignore
        if mol is None:
            return None

        clean_mol = rdMolStandardize.Cleanup(mol)
        return clean_mol
        # parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        # return parent_clean_mol
        # uncharger = rdMolStandardize.Uncharger()
        # uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        # return uncharged_parent_clean_mol


def mol_to_inchi(mol):
    """InChI for `mol`, or None if RDKit will not produce one.

    Three compounds in the MDR1-MDCK subset and three in the solubility subset
    survive `standardize` but carry rings RDKit cannot kekulize, and InChI
    generation raises on those instead of returning an empty string. Returning
    None lets callers drop them with the same `dropna` that handles SMILES
    which fail to parse.
    """
    with BlockLogs():
        try:
            return Chem.MolToInchi(mol)
        except Chem.rdchem.MolSanitizeException:
            return None


def taylor_butina_clustering(
    fp_list: list[DataStructs.ExplicitBitVect], cutoff: float = 0.65
) -> list[int]:
    """Cluster a set of fingerprints using the RDKit Taylor-Butina implementation

    :param fp_list: a list of fingerprints
    :param cutoff: distance cutoff (1 - Tanimoto similarity)
    :return: a list of cluster ids
    """
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    cluster_res = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_id_list = np.zeros(nfps, dtype=int)
    for cluster_num, cluster in enumerate(cluster_res):
        for member in cluster:
            cluster_id_list[member] = cluster_num
    return cluster_id_list.tolist()


def get_butina_clusters(mol_list, cutoff: float = 0.65) -> list[int]:
    """
    Cluster a list of SMILES strings using the Butina clustering algorithm.

    :param cutoff: The cutoff value to use for clustering
    :return: List of cluster labels corresponding to each SMILES string in the input list.
    """
    fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp_list = [fg.GetFingerprint(x) for x in mol_list]
    return taylor_butina_clustering(fp_list, cutoff=cutoff)


def process_dataset(df):
    df["mol"] = df["smiles"].map(standardize)
    df["inchi"] = df["mol"].map(mol_to_inchi)
    df = df.groupby(["inchi"]).filter(lambda x: len(x) == 1).reset_index(drop=True)
    df["mol"] = df["inchi"].map(Chem.MolFromInchi)

    df["butina_cluster"] = get_butina_clusters(df["mol"])
    return df
