import pandas as pd
import numpy as np
import pickle
from drevalpy.datasets.dataset import FeatureDataset
from ..utils import load_and_reduce_gene_features

def load_cell_line_features(data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Lädt die Genexpressionsmerkmale der Zelllinie unter Verwendung der bereitgestellten Genliste aus .pkl-Datei

    :param data_path: Pfad zum Datenverzeichnis
    :param dataset_name: Name des Datensatzes; GDSC2
    :returns: FeatureDataset mit reduzierten und gefilterten Merkmalen basierend auf der Genliste aus .pkl-Datei
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
