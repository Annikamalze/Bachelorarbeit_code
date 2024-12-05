import torch
import pandas as pd
from paccmann_smiles_embedding import SMILESEmbedding

def test_smiles_embedding():
    smiles_files = [
        'C:/Users/Annika/paccmann_predictor/data/smiles/ccle.smi',
        'C:/Users/Annika/paccmann_predictor/data/smiles/chembl_22_clean_1576904_sorted_std_final.smi',
        'C:/Users/Annika/paccmann_predictor/data/smiles/gdsc.smi'
    ]
    vocab_path = 'smiles_language_chembl_gdsc_ccle.pkl'
    embedding_size = 64
    
    # Modell instanziieren
    model = SMILESEmbedding(vocab_path, embedding_size, smiles_files=smiles_files)

    # Testen mit Beispiel-SMILES
    smiles_list = ["CCO", "CCN"]
    processed_smiles = model.preprocess_smiles(smiles_list)
    print("Processed SMILES:", processed_smiles)
    # Daten aus Datei:
    # with open('C:/Users/Annika/paccmann_predictor/data/smiles/deine_testdatei.smi', 'r') as f:
    # smiles_list = [line.strip() for line in f]
    
    # Speichern der Ergebnisse in eine CSV-Datei
    output_file = 'processed_smiles.csv'
    model.save_to_csv(processed_smiles, output_file)
    print(f"Processed SMILES saved to {output_file}")
    
    # Überprüfen der Einbettung
    embedded_smiles = model(processed_smiles)
    print("Embedded SMILES:", embedded_smiles)

# Test ausführen
if __name__ == '__main__':
    test_smiles_embedding()
