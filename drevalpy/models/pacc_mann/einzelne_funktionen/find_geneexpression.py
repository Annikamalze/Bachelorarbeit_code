import os

file_path = 'data/GDSC2/gene_expression/gene_expression.csv'
if os.path.exists(file_path):
    print(f"Die Datei existiert: {file_path}")
else:
    print(f"Die Datei existiert nicht: {file_path}")
