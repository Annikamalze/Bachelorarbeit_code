import torch
import pandas as pd
from paccmann_smiles_embedding import SMILESEmbedding

def test_smiles_embedding():
    # Pfad zur SMILES-Datei mit Namen
    #smiles_file = 'C:/Users/Annika/paccmann_predictor/data/smiles/gdsc.smi'
    smiles_file = 'C:/Users/Annika/paccmann_predictor/data/smiles/ccle.smi'
    # Pfad zur Vokabular-Datei und Größe der Embedding-Vektoren
    vocab_path = 'smiles_language_chembl_gdsc_ccle.pkl'
    embedding_size = 64
    
    # Modell initialisieren
    model = SMILESEmbedding(vocab_path, embedding_size)

    # Lesen der Datei und Extrahieren von SMILES und Namen
    smiles_data = []
    with open(smiles_file, 'r') as f:
        for line in f:
            # Annahme: Jede Zeile ist im Format "SMILES Name"
            parts = line.strip().split('\t')  # Tab als Trennzeichen
            if len(parts) == 2:
                smiles, name = parts
                smiles_data.append((name, smiles))
    
    # Verarbeiten der SMILES
    names = [entry[0].capitalize() for entry in smiles_data]
    smiles_list = [entry[1] for entry in smiles_data]
    processed_smiles = model.preprocess_smiles(smiles_list)
    
    # Speichern der verarbeiteten SMILES mit Namen in eine CSV-Datei
    output_file = 'processed_smiles_ccle_with_names.csv'
    processed_df = pd.DataFrame(processed_smiles.numpy())
    processed_df.insert(0, "Name", names)  # Namen als erste Spalte hinzufügen
    processed_df.to_csv(output_file, index=False)
    print(f"Processed SMILES with names saved to {output_file}")

if __name__ == '__main__':
    test_smiles_embedding()
