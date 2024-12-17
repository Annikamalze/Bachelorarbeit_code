import os
import pandas as pd
from typing import Optional
from drevalpy.datasets.dataset import FeatureDataset

class PaccMann:
    """
    Klasse zum Laden der Medikamentenmerkmale (SMILES).
    """

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

    def load_drug_features(self, data_path: str, dataset_name: str) -> Optional[FeatureDataset]:
        """
        Lädt die Medikamentenmerkmale, in diesem Fall die SMILES, für das angegebene Dataset.

        :param data_path: Pfad zu den Daten, z.B. "data/"
        :param dataset_name: Name des Datasets, z.B. "GDSC2"
        :returns: FeatureDataset mit den Medikamenten-SMILES
        """
        return self.load_drug_smiles_features(data_path, dataset_name)

if __name__ == "__main__":
    data_path = "./data/GDSC2/processed_smiles_gdsc2.csv"
    dataset_name = "GDSC2"
    model = PaccMann()
    
    try:
        feature_dataset = model.load_drug_features(data_path, dataset_name)
        print("Daten erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
