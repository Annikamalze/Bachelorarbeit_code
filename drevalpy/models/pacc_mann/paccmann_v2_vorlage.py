"""
Contains the PaccMann model.

"""

import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
import yaml
from drevalpy.models.drp_model import DRPModel
from ..utils import load_and_reduce_gene_features
from drevalpy.datasets.dataset import FeatureDataset
from model_utils import load_drug_smiles_features

class PaccMann(DRPModel):
    """
    Abstract wrapper class for drug response prediction models.

    The DRPModel class is an abstract wrapper class for drug response prediction models.
    It has a boolean attribute is_single_drug_model indicating whether it is a single drug model and a boolean
    attribute early_stopping indicating whether early stopping is used.
    """
    cell_line_views = ["gene_expression"]
    drug_views = ["SMILES"]

    def __init__(self):
        """
        Initializes the model.

        Sets the model to None, which is initialized in the build_model method to the respective sklearn model.
        """
        super().__init__()
        self.model = None

    # Used in the pipeline!
    early_stopping = False
    # Then, the model is trained per drug
    is_single_drug_model = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the name of the model.

        :return: model name
        """
        return "PaccMann model"
        
    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: includes units_per_layer and dropout_prob.
        """
        self.hyperparameters = hyperparameters

    @abstractmethod
    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load the cell line features before the train/predict method is called.

        Required to implement for all models. Could, e.g., call get_multiomics_feature_dataset() or
        load_and_reduce_gene_features() from models/utils.py.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., "GDSC2"
        :returns: FeatureDataset with the cell line features
        """
        return load_and_reduce_gene_features(
        feature_type="gene_expression",
        gene_list="2128_genes", 
        data_path=data_path,
        dataset_name=dataset_name,
    )

    @abstractmethod
    def load_drug_features(self, data_path: str, dataset_name: str) -> Optional[FeatureDataset]:
        """
        Load the drug features before the train/predict method is called.

        Required to implement for all models that use drug features. Could, e.g.,
        call load_drug_fingerprint_features() or load_drug_ids_from_csv() from models/utils.py.

        For single drug models, this method can return None.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., "GDSC2"
        :returns: FeatureDataset or None
        """
        return self.load_drug_smiles_features(data_path, dataset_name)
    

# Testfunktionen
def test_load_cell_line_features():
    data_path = "data/GDSC2"
    dataset_name = "gene_expression"
    gene_list_path = os.path.join(data_path, "2128_genes.csv")
    
    print("Teste Laden der Zelllinienmerkmale...")
    try:
        genes = load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list=gene_list_path,
            data_path=data_path,
            dataset_name=dataset_name
        )
        if genes:
            print(f"Erfolgreich Gene geladen: {len(genes)} Gene.")
        else:
            print("Keine Gene geladen.")
    except Exception as e:
        print(f"Fehler beim Laden der Zelllinienmerkmale: {e}")

def test_load_drug_features():
    data_path = "data/GDSC2"
    dataset_name = "GDSC2"
    
    print("Teste Laden der Wirkstoffmerkmale...")
    try:
        model = PaccMann()
        feature_dataset = model.load_drug_features(data_path, dataset_name)
        if feature_dataset:
            print("Wirkstoffmerkmale erfolgreich geladen.")
        else:
            print("Keine Wirkstoffmerkmale geladen.")
    except Exception as e:
        print(f"Fehler beim Laden der Wirkstoffmerkmale: {e}")

# Main
if __name__ == "__main__":
    test_load_cell_line_features()
    test_load_drug_features()
