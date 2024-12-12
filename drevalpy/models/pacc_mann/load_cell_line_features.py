import pandas as pd
import os
from drevalpy.datasets.dataset import FeatureDataset
from utils import load_and_reduce_gene_features

def load_cell_line_features(data_path: str, dataset_name: str, gene_list_path: str):
    """
    Lädt die Genexpressionsmerkmale der Zelllinie unter Verwendung der Genliste aus CSV-Datei
    
    :param data_path: Pfad zum Datenverzeichnis
    :param dataset_name: Name des Datensatzes; GDSC2
    :param gene_list_path: Pfad zur CSV-Datei mit der Genliste
    :returns: FeatureDataset mit reduzierten und gefilterten Merkmalen
    """
    return load_and_reduce_gene_features(
        feature_type="gene_expression",
        gene_list="2128_genes", 
        data_path=data_path,
        dataset_name=dataset_name,
    )

def test_load_cell_line_features():
    data_path = "data/GDSC2"
    dataset_name = "gene_expression"
    gene_list_path = os.path.join(data_path, "2128_genes.csv")  

    try:
        genes = load_cell_line_features(data_path, dataset_name, gene_list_path)
        if genes:
            print(f"Erfolgreich {len(genes)} Gene geladen.")
        else:
            print("Keine Gene geladen.")
    except Exception as e:
        print("Es gab einen Fehler:", e)

if __name__ == "__main__":
    test_load_cell_line_features()
