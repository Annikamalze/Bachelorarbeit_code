import os
from abc import ABC
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..utils import load_and_reduce_gene_features
from drevalpy.datasets.dataset import FeatureDataset

def load_specific_genes_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Lädt die Merkmale der spezifischen Gene.

    :param data_path: Pfad zu den Genexpressionsdaten und den spezifischen Genen
    :param dataset_name: Name des Datensatzes
    :returns: FeatureDataset mit den Genexpressionsmerkmalen der spezifischen Gene
    """
    return load_and_reduce_gene_features(
        feature_type="gene_expression",
        gene_list="2128_genes.pkl",  # Hier wird die Liste der spezifischen Gene verwendet 2128_genes.pkl
        data_path=data_path,
        dataset_name=dataset_name,
    )