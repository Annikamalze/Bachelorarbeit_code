import pytest
import numpy as np

from drevalpy.evaluation import evaluate, pearson
from drevalpy.models import MODEL_FACTORY
from .utils import call_save_and_load, sample_dataset


@pytest.mark.parametrize("test_mode", ["LCO"])
@pytest.mark.parametrize("model_name", ["MOLIR"])
def test_molir(sample_dataset, model_name, test_mode):
    drug_response, cell_line_input, drug_input = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_es_dataset = split["validation_es"]
    es_dataset = split["early_stopping"]
    all_predictions = np.zeros_like(val_es_dataset.drug_ids, dtype=float)
    all_unique_drugs = np.unique(train_dataset.drug_ids)
    # randomly sample 10
    np.random.seed(42)
    np.random.shuffle(all_unique_drugs)
    all_unique_drugs = all_unique_drugs[:10]
    for drug in all_unique_drugs:
        model = MODEL_FACTORY[model_name]()
        hpam_combi = model.get_hyperparameter_set()[0]
        model.build_model(hpam_combi)

        output_mask = train_dataset.drug_ids == drug
        drug_train = train_dataset.copy()
        drug_train.mask(output_mask)
        es_mask = es_dataset.drug_ids == drug
        es_dataset_drug = es_dataset.copy()
        es_dataset_drug.mask(es_mask)

        model.train(
            output=drug_train,
            cell_line_input=cell_line_input,
            drug_input=None,
            output_earlystopping=es_dataset_drug,
        )

        val_mask = val_es_dataset.drug_ids == drug
        all_predictions[val_mask] = model.predict(
            drug_ids=drug,
            cell_line_ids=val_es_dataset.cell_line_ids[val_mask],
            cell_line_input=cell_line_input,
        )
        pcc_drug = pearson(val_es_dataset.response[val_mask], all_predictions[val_mask])
        print(f"{test_mode}: Performance of {model_name} for drug {drug}: PCC = {pcc_drug}")
    val_es_dataset.predictions = all_predictions
    metrics = evaluate(val_es_dataset, metric=["Pearson"])
    print(f"{test_mode}: Collapsed performance of {model_name}: PCC = {metrics['Pearson']}")
    assert metrics["Pearson"] > 0.0
