import os
import pandas as pd
from typing import Optional
from drevalpy.datasets.dataset import FeatureDataset


def load_drug_smiles_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Lädt die SMILES-Merkmale für das angegebene Dataset.

        :param data_path: Pfad zu den Daten, z.B. "data/"
        :param dataset_name: Name des Datasets, z.B. "GDSC2"
        :returns: FeatureDataset mit den Medikamenten-SMILES
        """
        if dataset_name == "GDSC2":
            smiles_file = "processed_smiles_gdsc2.csv" 
        elif dataset_name == "CCLE":
            smiles_file = "processed_smiles_ccle.csv"  
        else:
            raise ValueError(f"Unbekanntes Dataset: {dataset_name}")

        smiles = pd.read_csv(os.path.join(data_path, dataset_name, smiles_file), index_col=0)

        return FeatureDataset(
        features={drug: {"SMILES": smiles.loc[drug].values} for drug in smiles.index}
    )