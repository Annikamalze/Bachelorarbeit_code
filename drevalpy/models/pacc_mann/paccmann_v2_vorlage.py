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

class PaccMann(DRPModel):
    """
    Abstract wrapper class for drug response prediction models.

    The DRPModel class is an abstract wrapper class for drug response prediction models.
    It has a boolean attribute is_single_drug_model indicating whether it is a single drug model and a boolean
    attribute early_stopping indicating whether early stopping is used.
    """

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

    @classmethod
    def get_hyperparameter_set(cls) -> list[dict[str, Any]]:
        """
        Loads the hyperparameters from a yaml file which is located in the same directory as the model.

        :returns: list of hyperparameter sets
        :raises ValueError: if the hyperparameters are not in the correct format
        :raises KeyError: if the model is not found in the hyperparameters file
        """
        hyperparameter_file = os.path.join(os.path.dirname(inspect.getfile(cls)), "hyperparameters.yaml")

        with open(hyperparameter_file, encoding="utf-8") as f:
            try:
                hpams = yaml.safe_load(f)[cls.get_model_name()]
            except yaml.YAMLError as exc:
                raise ValueError(f"Error in hyperparameters.yaml: {exc}") from exc
            except KeyError as key_exc:
                raise KeyError(f"Model {cls.get_model_name()} not found in hyperparameters.yaml") from key_exc

        if hpams is None:
            return [{}]
        # each param should be a list
        for hp in hpams:
            if not isinstance(hpams[hp], list):
                hpams[hp] = [hpams[hp]]
        grid = list(ParameterGrid(hpams))
        return grid

    @property
    @abstractmethod
    def cell_line_views(self) -> list[str]:
        """
        Returns the sources the model needs as input for describing the cell line.

        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression",
            "mutation"]. If the model does not use cell line features, return an empty list.
        """

    @property
    @abstractmethod
    def drug_views(self) -> list[str]:
        """
        Returns the sources the model needs as input for describing the drug.

        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]. If the model does not use drug features,
            return an empty list.
        """

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