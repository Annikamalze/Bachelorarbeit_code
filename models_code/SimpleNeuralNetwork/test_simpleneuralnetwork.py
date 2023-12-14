# TODO this is not a proper test, but rather example/experimentation script
from models_code import SimpleNeuralNetwork
from suite.data_wrapper import DrugResponseDataset
import pandas as pd

neural_net_baseline = SimpleNeuralNetwork("smpl", target="IC50")

models = [neural_net_baseline]
nn_hpam_set = [
    {"dropout_prob": 0.2, "units_per_layer": [10, 10, 10]},
    {"dropout_prob": 0.3, "units_per_layer": [20, 10, 10]},
]
hpam_sets = [nn_hpam_set]
feature_path = "data/GDSC/"  # maybe this should be a parameter of the model class, so that the model can load the features itself, but also depends on the response dataset :S
for model, model_hpam_set in zip(models, hpam_sets):
    cl_features = model.get_cell_line_features(path=feature_path)
    drug_features = model.get_drug_features(path=feature_path)

    response_data = pd.read_csv("data/GDSC/response_GDSC2.csv")
    output = response_data["LN_IC50"].values
    cell_line_ids = response_data["CELL_LINE_NAME"].values
    drug_ids = response_data["DRUG_NAME"].values
    response_data = DrugResponseDataset(
        target_type="IC50",
        response=output,
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
    )

    # making sure there are no missing features:
    response_data.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )
    # todo crossvalidation splits etc

    for hyperparameter in model_hpam_set:
        model.train(
            cell_line_input=cl_features,
            drug_input=drug_features,
            output=response_data,
            hyperparameters=hyperparameter,
        )
        output = model.predict(cell_line_input=cl_features, drug_input=drug_features)

        # eval...
