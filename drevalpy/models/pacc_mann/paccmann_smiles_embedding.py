import torch
import torch.nn as nn
import pickle
import os
import pandas as pd

class SMILESEmbedding(nn.Module):
    """SMILES Einbettung

    Code adapted from the
    PaccMann model https://github.com/PaccMann/paccmann_predictor.
    """

    def _init_(self, smiles_vocabulary_path, smiles_embedding_size, smiles_files=None, scale_grad_by_freq=False):
        """
        SMILES-Einbettung
        :param smiles_vocabulary_path: Pfad zur Vokabel-Datei 
        :param smiles_embedding_size: Größe der Embedding-Vektoren
        :param smiles_files: Optional: Liste von SMILES-Dateien, um Vokabular zu erstellen
        :param scale_grad_by_freq: Skalierung der Gradienten basierend auf Häufigkeit. Default = False
        """
        super(SMILESEmbedding, self)._init_()

        if not os.path.exists(smiles_vocabulary_path):
            if smiles_files is None:
                raise ValueError("SMILES-Dateien müssen angegeben werden, um Vokabular zu erstellen.")
            self.smiles_vocab = self.create_vocab(smiles_files)

            with open(smiles_vocabulary_path, 'wb') as f:
                pickle.dump(self.smiles_vocab, f)
        else:
            self.smiles_vocab = self.load_vocab(smiles_vocabulary_path)
        
        self.smiles_vocabulary_size = len(self.smiles_vocab)

        self.smiles_embedding = nn.Embedding(
            self.smiles_vocabulary_size,
            smiles_embedding_size,
            scale_grad_by_freq=scale_grad_by_freq
        )

    def load_vocab(self, vocab_path):
        """
        Lade das SMILES-Vokabular aus .pkl-Datei
        :param vocab_path: Pfad zur Vokabel-Datei
        :return: Wörterbuch mit {Token: Index}
        """
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vokabular geladen. Größe: {len(vocab)}")
        return vocab

    def create_vocab(self, smiles_files):
        """
        Erstelle das Vokabular aus angegebenen SMILES-Dateien
        :param smiles_files: Liste der SMILES-Dateien
        :return: Wörterbuch mit {Token: Index}
        """
        vocab = set()
        for smiles_file in smiles_files:
            with open(smiles_file, 'r') as f:
                for line in f:
                    for char in line.strip():  
                        vocab.add(char)
        
        # Erstelle das Vokabular-Dictionary (Token -> Index)
        vocab_dict = {token: idx for idx, token in enumerate(sorted(vocab))}
        print(f"Vokabular erstellt. Größe: {len(vocab_dict)}")
        return vocab_dict

    def tokenize_and_integerize(self, smiles):
        """
        Tokenisiere eine SMILES-Zeichenkette und wandle in Integer um
        :param smiles: SMILES-Zeichenkette
        :return: Liste von Indizes
        """
        tokens = list(smiles)  
        return [self.smiles_vocab[token] for token in tokens if token in self.smiles_vocab]

    def preprocess_smiles(self, smiles_list):
        """
        Preprocessing von SMILES-Zeichenketten
        :param smiles_list: Liste von SMILES-Zeichenketten
        :return: Tensor mit integerisierten SMILES
        """
        processed = [self.tokenize_and_integerize(smiles) for smiles in smiles_list]
        max_len = max(len(smiles) for smiles in processed)
        
        padded = [smiles + [0] * (max_len - len(smiles)) for smiles in processed]
        return torch.tensor(padded, dtype=torch.long)

    def save_to_csv(self, processed_smiles, file_path):
        """
        Speichert verarbeitete SMILES in CSV-Datei
        :param processed_smiles: Tensor mit integerisierten und gepaddeten SMILES
        :param file_path: Pfad zur Ausgabedatei
        """
        # Konvertiere den Tensor und speichere ihn als DataFrame
        df = pd.DataFrame(processed_smiles.numpy())
        df.to_csv(file_path, index=False)
        print(f"SMILES-Daten wurden in {file_path} gespeichert.")

    def forward(self, smiles_indices):
        """
        Führt Einbetten für gegebene SMILES-Indizes aus
        :param smiles_indices: Tensor mit den integerisierten SMILES-Indizes
        :return: Tensor mit den eingebetteten SMILES
        """
        return self.smiles_embedding(smiles_indices)
