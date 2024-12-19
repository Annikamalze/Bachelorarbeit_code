import torch
import pandas as pd
from paccmann_smiles_embedding import SMILESEmbedding

def test_smiles_embedding():

    # smiles_files = [
    #     'C:/Users/Annika/Bachelorarbeit_coda/data/ccle.smi',
    #     'C:/Users/Annika/paccmann_predictor/data/smiles/chembl_22_clean_1576904_sorted_std_final.smi',
    #     'C:/Users/Annika/paccmann_predictor/data/smiles/gdsc.smi'
    # ]
    
    vocab_path = 'smiles_language_chembl_gdsc_ccle.pkl'
    embedding_size = 64
    
    model = SMILESEmbedding(vocab_path, embedding_size)  # Ohne smiles_files

    #smiles_file = 'C:/Users/Annika/Bachelorarbeit_code/data/ccle.smi'
    smiles_file = 'C:/Users/Annika/paccmann_predictor/data/smiles/gdsc.smi'
    with open(smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f]

    processed_smiles = model.preprocess_smiles(smiles_list)

    output_file = 'processed_smiles_gdsc.csv'
    model.save_to_csv(processed_smiles, output_file)
    
    embedded_smiles = model(processed_smiles)
    #print("Embedded SMILES:", embedded_smiles)

if __name__ == '__main__':
    test_smiles_embedding()
