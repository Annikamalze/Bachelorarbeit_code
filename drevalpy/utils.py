"""Utility functions for the evaluation pipeline."""

import argparse

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from drevalpy.datasets import RESPONSE_DATASET_FACTORY
from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.experiment import drug_response_experiment
from drevalpy.models import MODEL_FACTORY


def get_parser():
    """
    Get the parser for the evaluation pipeline.
    :return:
    """
    parser = argparse.ArgumentParser(description="Run the drug response prediction model test suite.")
    parser.add_argument(
        "--run_id",
        type=str,
        default="my_run",
        help="identifier to save the results",
    )

    parser.add_argument(
        "--path_data",
        type=str,
        default="data",
        help="Path to the data directory",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="model to evaluate or list of models to compare",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        help="baseline or list of baselines. The baselines are also hpam-tuned and compared to the "
        "models, but no randomization or robustness tests are run.",
    )
    parser.add_argument(
        "--test_mode",
        nargs="+",
        default=["LPO"],
        help="Which tests to run (LPO=Leave-random-Pairs-Out, "
        "LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out). Can be a list of test runs e.g. "
        "'LPO LCO LDO' to run all tests. Default is LPO",
    )
    parser.add_argument(
        "--randomization_mode",
        nargs="+",
        default=["None"],
        help="Which randomization tests to run, additionally to the normal run. Default is None "
        "which means no randomization tests are run."
        "Modes: SVCC, SVRC, SVCD, SVRD"
        "Can be a list of randomization tests e.g. 'SCVC SCVD' to run two tests. Default is None"
        "SVCC: Single View Constant for Cell Lines: in this mode, one experiment is done for every "
        "cell line view the model uses (e.g. gene expression, mutation, ..)."
        "For each experiment one cell line view is held constant while the others are randomized. "
        "SVRC Single View Random for Cell Lines: in this mode, one experiment is done for every "
        "cell line view the model uses (e.g. gene expression, mutation, ..)."
        "For each experiment one cell line view is randomized while the others are held constant."
        "SVCD: Single View Constant for Drugs: in this mode, one experiment is done for every "
        "drug view the model uses (e.g. fingerprints, target_information, ..)."
        "For each experiment one drug view is held constant while the others are randomized."
        "SVRD: Single View Random for Drugs: in this mode, one experiment is done for every "
        "drug view the model uses (e.g. gene expression, target_information, ..)."
        "For each experiment one drug view is randomized while the others are held constant.",
    )
    parser.add_argument(
        "--randomization_type",
        type=str,
        default="permutation",
        help='type of randomization to use. Choose from "permutation" or "invariant". Default is '
        '"permutation" "permutation": permute the features over the instances, keeping the '
        "distribution of the  features the same but dissolving the relationship to the "
        'target "invariant": the randomization is done in a way that a key characteristic of '
        "the feature is preserved. In case of matrices, this is the mean and standard "
        "deviation of the feature view for this instance, for networks it is the degree "
        "distribution.",
    )
    parser.add_argument(
        "--n_trials_robustness",
        type=int,
        default=0,
        help="Number of trials to run for the robustness test. Default is 0, which means no "
        "robustness test is run. The robustness test is a test where the model is trained "
        "with varying seeds. This is done multiple times to see how stable the model is.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="GDSC1",
        help="Name of the drug response dataset",
    )

    parser.add_argument(
        "--cross_study_datasets",
        nargs="+",
        default=[],
        help="List of datasets to use to evaluate predictions across studies. Default is empty list"
        " which means no cross-study datasets are used.",
    )

    parser.add_argument(
        "--path_out",
        type=str,
        default="results/",
        help="Path to the output directory",
    )

    parser.add_argument(
        "--curve_curator",
        action="store_true",
        default=False,
        help="Whether to run " "CurveCurator " "to sort out " "non-reactive " "curves",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing results with the same path out and run_id? ",
    )
    parser.add_argument(
        "--optim_metric",
        type=str,
        default="RMSE",
        help=f"Metric for hyperparameter tuning choose from {list(AVAILABLE_METRICS.keys())} " f"Default is RMSE.",
    )
    parser.add_argument(
        "--n_cv_splits",
        type=int,
        default=5,
        help="Number of cross-validation splits to use for the evaluation",
    )

    parser.add_argument(
        "--response_transformation",
        type=str,
        default="None",
        help="Transformation to apply to the response variable during training and prediction. "
        "Will be retransformed after the final predictions. Possible values: standard, "
        "minmax, robust",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        default=False,
        help="Whether to use multiprocessing for the evaluation. Default is False",
    )

    return parser


def check_arguments(args):
    """
    Check the validity of the arguments for the evaluation pipeline.

    :param args:
    :return:
    """
    if not args.models:
        raise AssertionError("At least one model must be specified")
    if not all(model in MODEL_FACTORY for model in args.models):
        raise AssertionError(
            f"Invalid model name. Available models are {list(MODEL_FACTORY.keys())}. If you want to "
            f"use your own model, you need to implement a new model class and add it to the "
            f"MODEL_FACTORY in the models init"
        )
    if not all(test in ["LPO", "LCO", "LDO"] for test in args.test_mode):
        raise AssertionError("Invalid test mode. Available test modes are LPO, LCO, LDO")

    if args.baselines is not None:
        if not all(baseline in MODEL_FACTORY for baseline in args.baselines):
            raise AssertionError(
                f"Invalid baseline name. Available baselines are {list(MODEL_FACTORY.keys())}. If you "
                f"want to use your own baseline, you need to implement a new model class and add it to "
                f"the MODEL_FACTORY in the models init"
            )

    if args.dataset_name not in RESPONSE_DATASET_FACTORY:
        raise AssertionError(
            f"Invalid dataset name. Available datasets are {list(RESPONSE_DATASET_FACTORY.keys())} "
            f"If you want to use your own dataset, you need to implement a new response dataset class "
            f"and add it to the RESPONSE_DATASET_FACTORY in the response_datasets init"
        )

    for dataset in args.cross_study_datasets:
        if dataset not in RESPONSE_DATASET_FACTORY:
            raise AssertionError(
                f"Invalid dataset name in cross_study_datasets. Available datasets are "
                f"{list(RESPONSE_DATASET_FACTORY.keys())} If you want to use your own dataset, you "
                f"need to implement a new response dataset class and add it to the "
                f"RESPONSE_DATASET_FACTORY in the response_datasets init"
            )

    if args.n_cv_splits <= 1:
        raise ValueError("Number of cross-validation splits must be greater than 1")

    # TODO Allow for custom randomization tests maybe via config file
    if args.randomization_mode[0] != "None":
        if not all(randomization in ["SVCC", "SVRC", "SVSC", "SVRD"] for randomization in args.randomization_mode):
            raise AssertionError(
                "At least one invalid randomization mode. Available randomization modes are SVCC, " "SVRC, SVSC, SVRD"
            )
    if args.curve_curator:
        raise NotImplementedError("CurveCurator not implemented")
    if args.response_transformation not in ["None", "standard", "minmax", "robust"]:
        raise AssertionError("Invalid response_transformation. Choose from None, standard, minmax, robust")
    if args.optim_metric not in AVAILABLE_METRICS:
        raise AssertionError(
            f"Invalid optim_metric for hyperparameter tuning. Choose from" f" {list(AVAILABLE_METRICS.keys())}"
        )


def main(args):
    """
    Main function to run the drug response evaluation pipeline.
    :param args: passed from command line
    :return:
    """
    check_arguments(args)

    # PIPELINE: LOAD_RESPONSE
    response_data, cross_study_datasets = load_data(
        dataset_name=args.dataset_name,
        cross_study_datasets=args.cross_study_datasets,
        path_data=args.path_data,
    )

    models = [MODEL_FACTORY[model] for model in args.models]

    if args.baselines is not None:
        baselines = [MODEL_FACTORY[baseline] for baseline in args.baselines]
    else:
        baselines = []
    # TODO Allow for custom randomization tests maybe via config file

    if args.randomization_mode[0] == "None":
        args.randomization_mode = None
    response_transformation = get_response_transformation(args.response_transformation)

    for test_mode in args.test_mode:
        drug_response_experiment(
            models=models,
            baselines=baselines,
            response_data=response_data,
            response_transformation=response_transformation,
            metric=args.optim_metric,
            n_cv_splits=args.n_cv_splits,
            multiprocessing=args.multiprocessing,
            test_mode=test_mode,
            randomization_mode=args.randomization_mode,
            randomization_type=args.randomization_type,
            n_trials_robustness=args.n_trials_robustness,
            cross_study_datasets=cross_study_datasets,
            path_out=args.path_out,
            run_id=args.run_id,
            overwrite=args.overwrite,
            path_data=args.path_data,
        )


def load_data(dataset_name: str, cross_study_datasets: list, path_data: str = "data"):
    """
    Load the response data and cross-study datasets.
    :param dataset_name:
    :param cross_study_datasets:
    :param path_data:
    :return:
    """
    # PIPELINE: LOAD_RESPONSE
    response_data = RESPONSE_DATASET_FACTORY[dataset_name](path_data=path_data)

    cross_study_datasets = [RESPONSE_DATASET_FACTORY[dataset](path_data=path_data) for dataset in cross_study_datasets]
    return response_data, cross_study_datasets


def get_response_transformation(response_transformation: str):
    """
    Get the response transformation object.
    :param response_transformation:
    :return:
    """
    if response_transformation == "None":
        return None
    if response_transformation == "standard":
        return StandardScaler()
    if response_transformation == "minmax":
        return MinMaxScaler()
    if response_transformation == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown response transformation {response_transformation}. Choose from 'None', " f"'standard', 'minmax', 'robust'")
