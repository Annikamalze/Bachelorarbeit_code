import pandas as pd
import numpy as np
import pickle
from drevalpy.datasets.dataset import FeatureDataset
from ..utils import load_and_reduce_gene_features


def load_cell_line_features(data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Loads the cell line gene expression features using provided gene list from a pickle file.

    :param data_path: Path to the data directory.
    :param dataset_name: Name of the dataset, e.g., GDSC2.
    :returns: FeatureDataset with reduced and filtered features based on the gene list from the pickle file.
    """
    
    # Lade die Gene aus der 2128_genes.pkl Datei
    gene_list_path = f"{data_path}/{dataset_name}/2128_genes.pkl"
    
    with open(gene_list_path, "rb") as f:
        genes = pickle.load(f)  # Lade die Genliste aus der Pickle-Datei

    # Lade und reduziere die Gene mit der Hilfsfunktion
    return load_and_reduce_gene_features(
        feature_type="gene_expression",
        gene_list=genes, 
        data_path=data_path,
        dataset_name=dataset_name,
    )
