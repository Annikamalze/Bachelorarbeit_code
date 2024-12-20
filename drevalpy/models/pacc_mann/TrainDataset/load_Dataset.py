import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Dateien laden
gene_expression_path = "/storage/mi/malzea03/Bachelorarbeit_code/data/GDSC2/gene_expression/gene_expression.csv"
smiles_data_path = "/storage/mi/malzea03/Bachelorarbeit_code/data/processed_smiles_GDSC2.csv"
drug_sensitivity_path = "/storage/mi/malzea03/Bachelorarbeit_code/data/GDSC2/gdsc_cell_line_name_drug_sensitivity.csv"

# Dateien einlesen
gene_expression = pd.read_csv(gene_expression_path)
smiles_data = pd.read_csv(smiles_data_path)
drug_sensitivity = pd.read_csv(drug_sensitivity_path)

# Überprüfen der ersten paar Zeilen und Spaltennamen der geladenen DataFrames
print("Gene Expression Columns:", gene_expression.columns)
print("SMILES Data Columns:", smiles_data.columns)
print("Drug Sensitivity Columns:", drug_sensitivity.columns)

# Daten prüfen und anpassen
gene_expression.rename(columns={"cellosaurus_id": "cell_line"}, inplace=True)

# Verknüpfen von Drug Sensitivity mit Gene Expression
merged_data = pd.merge(drug_sensitivity, gene_expression, on="cell_line", how="inner")
print(f"Nach dem Zusammenführen mit Gene Expression: {merged_data.shape} Zeilen und Spalten")

# Verknüpfen von SMILES-Daten
merged_data = pd.merge(merged_data, smiles_data, left_on="drug", right_on="Name", how="inner")
print(f"Nach dem Zusammenführen mit SMILES: {merged_data.shape} Zeilen und Spalten")

# Überprüfen, ob IC50-Werte vorhanden sind
if "IC50" in merged_data.columns:
    print("IC50 Spalte gefunden!")
    ic50 = merged_data["IC50"].values
else:
    print("IC50 Spalte nicht gefunden!")
    ic50 = None

# Gene-Expression-Daten (Annahme: Spalten mit "gene_" beginnen)
gene_expr_columns = [col for col in merged_data.columns if col.startswith("gene_")]
print(f"Gene Expression Columns: {gene_expr_columns}")
gene_expr = merged_data[gene_expr_columns].values if gene_expr_columns else None

# SMILES-Daten (Annahme: Spalten mit "smiles_" beginnen)
smiles_columns = [col for col in merged_data.columns if col.startswith("smiles_")]
print(f"SMILES Columns: {smiles_columns}")
smiles = merged_data[smiles_columns].values if smiles_columns else None

# Überprüfen, ob die extrahierten Daten vorhanden sind
print(f"IC50 Shape: {ic50.shape if ic50 is not None else 'None'}")
print(f"Gene Expression Shape: {gene_expr.shape if gene_expr is not None else 'None'}")
print(f"SMILES Shape: {smiles.shape if smiles is not None else 'None'}")

# Daten umwandeln und prüfen
if ic50 is not None and gene_expr is not None and smiles is not None:
    ic50_tensor = torch.tensor(ic50, dtype=torch.float32)
    gene_expr_tensor = torch.tensor(gene_expr, dtype=torch.float32)
    smiles_tensor = torch.tensor(smiles, dtype=torch.float32)

    # TensorDataset erstellen
    dataset = TensorDataset(smiles_tensor, gene_expr_tensor, ic50_tensor)

    # DataLoader für Batch-Training
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test: Daten iterieren
    for batch in data_loader:
        smiles_batch, gene_expr_batch, ic50_batch = batch
        print(f"SMILES Shape: {smiles_batch.shape}, Gene Expr Shape: {gene_expr_batch.shape}, IC50 Shape: {ic50_batch.shape}")
        break  # Nur den ersten Batch anzeigen
else:
    print("Daten fehlen: SMILES, Gene Expressions oder IC50 konnten nicht korrekt extrahiert werden.")
