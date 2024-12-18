from hilfsfunktionen.paccmann_utils import PaccMannV2

import torch
from torch.optim import Adam

class MyModel:
    def __init__(self):
        self.hyperparameters = None
        self.model = None

    def build_model(self, hyperparameters: dict):
        """
        Baut das Modell basierend auf den Hyperparametern auf.

        :param hyperparameters: Dictionary mit Parametern wie units_per_layer und dropout_prob.
        """
        self.hyperparameters = hyperparameters
        # Beispielmodellaufbau
        # self.model = SomeModel(
        #     layers=hyperparameters["units_per_layer"],
        #     dropout=hyperparameters["dropout_prob"]
        # )
        pass

    def forward(self, smiles, gep):
        """
        Beispiel für eine Vorwärtspropagation (anpassbar).
        """
        # Implementiere die Vorwärtspropagation des Modells
        pass

    def loss(self, predictions, targets):
        """
        Verlustfunktion.
        """
        return torch.nn.MSELoss()(predictions, targets)

def train_model(model, train_loader, params, device):
    """
    Trainiert das Modell über mehrere Epochen.

    :param model: Das zu trainierende Modell.
    :param train_loader: DataLoader für die Trainingsdaten.
    :param params: Dictionary mit Trainingsparametern wie "epochs".
    :param device: Zielgerät ("cuda" oder "cpu").
    """
    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    
    for epoch in range(params["epochs"]):
        model.train()

        for smiles, gep, y in train_loader:
            smiles, gep, y = smiles.to(device), gep.to(device), y.to(device)
            y_hat, pred_dict = model(torch.squeeze(smiles), gep)
            loss = model.loss(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Hauptausführung
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hyperparameter importieren
    params = hyperparameters
    
    # Modell initialisieren
    model = MyModel()
    model.build_model(params)
    
    # Beispiel-DataLoader (Dummy-Daten)
    train_loader = torch.utils.data.DataLoader(
        dataset=[(torch.rand(1, 10), torch.rand(10), torch.rand(1)) for _ in range(100)],
        batch_size=16
    )
    
    # Training
    train_model(model, train_loader, params, device)