from drevalpy import load_cell_line_features  # Importiere die Funktion aus deinem Code

# Definiere Testpfade
data_path = "data/GDSC2"
dataset_name = "gene_expression"
gene_list_path = "data/2128_genes.pkl"

# Teste die Ladefunktion
try:
    features = load_cell_line_features(data_path, dataset_name, gene_list_path)
    print("Features geladen:", features)
except Exception as e:
    print("Es gab einen Fehler:", e)
