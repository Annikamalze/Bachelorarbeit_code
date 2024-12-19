import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Deine Daten einlesen
gene_expression = pd.read_csv("/storage/mi/malzea03/Bachelorarbeit_code/data/GDSC2/gene_expression/gene_expression.csv")
smiles_data = pd.read_csv("/storage/mi/malzea03/Bachelorarbeit_code/data/processed_smiles_GDSC2.csv")
response_data = pd.read_csv("/storage/mi/malzea03/Bachelorarbeit_code/data/GDSC2/response_GDSC2.csv")

# Verknüpfung der Daten: Zelllinien-ID als Schlüssel
gene_expression = gene_expression.set_index('cellosaurus_id')
smiles_data = smiles_data.set_index('Name')
response_data = response_data.set_index('cellosaurus_id')

# Merging der Gene Expression mit den SMILES-Daten und den Antworten (Response)
# Annahme: Die Zelllinien sind der gemeinsame Schlüssel
merged_data = gene_expression.join(smiles_data, on='cellosaurus_id').join(response_data['LN_IC50'], on='cellosaurus_id')

# Vorbereitung der Daten
X_gene_expression = merged_data.drop(columns=['LN_IC50']).values  # Gene Expression
X_smiles = merged_data.drop(columns=['LN_IC50']).iloc[:, -154:].values  # Gepaddete SMILES-Daten (154 features)
y_response = merged_data['LN_IC50'].values  # IC50 als Antwortvariable

# TensorDataset erstellen
dataset = torch.utils.data.TensorDataset(torch.tensor(X_gene_expression, dtype=torch.float32), 
                                         torch.tensor(X_smiles, dtype=torch.float32), 
                                         torch.tensor(y_response, dtype=torch.float32))

# DataLoader erstellen
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Beispiel: Iteriere durch den DataLoader und gib die Batchgrößen aus
for batch in train_loader:
    gene_exp_batch, smiles_batch, response_batch = batch
    print(gene_exp_batch.shape, smiles_batch.shape, response_batch.shape)
