import requests
import zipfile
import os
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import networkx as nx


def download_dataset(
    dataset: str,
    data_path: str = "data",
    record_id: str = 12633988,
    redownload: bool = False,
):
    file_name = f"{dataset}.zip"
    file_path = os.path.join(data_path, file_name)
    if os.path.exists(file_path) and not redownload:
        print(f"{dataset} already exists, skipping download.")
    else:
        # Zenodo API URL
        url = f"https://zenodo.org/api/records/{record_id}"

        # Fetch the record metadata
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Error fetching record: {response.status_code}")
        data = response.json()

        # Ensure the save path exists
        os.makedirs(data_path, exist_ok=True)

        # Download each file
        name_to_url = {file["key"]: file["links"]["self"] for file in data["files"]}
        file_url = name_to_url[file_name]
        # Download the file
        print(f"Downloading {dataset} from {file_url}...")
        response = requests.get(file_url)
        if response.status_code != 200:
            raise Exception(f"Error downloading file {dataset}: {response.status_code}")

        # Save the file
        with open(file_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(data_path)
        os.remove(file_path)  # Remove zip file after extraction

        print(f"CCLE data downloaded and extracted to {data_path}")




def randomize_graph(original_graph: nx.Graph) -> nx.Graph:
    """
    Randomizes the graph by shuffling the edges while preserving the degree sequence.
    :param original_graph: The original graph
    :return: Randomized graph with the same degree sequence and node attributes
    """
    # Get the degree sequence from the original graph
    degree_sequence = [degree for node, degree in original_graph.degree()]

    # Generate a new graph with the expected degree sequence
    new_graph = nx.expected_degree_graph(degree_sequence, seed=1234)

    # Remap nodes to the original labels
    mapping = {new_node: old_node for new_node, old_node in zip(new_graph.nodes(), original_graph.nodes())}
    new_graph = nx.relabel_nodes(new_graph, mapping)

    # Copy node attributes from the original graph to the new graph
    for node, data in original_graph.nodes(data=True):
        new_graph.nodes[node].update(data)

    # Get the edge attributes from the original graph
    edge_attributes = list(original_graph.edges(data=True))

    # Assign random edge attributes to the new edges
    for edge in new_graph.edges():
        _, _, attr = edge_attributes[np.random.randint(len(edge_attributes))]
        new_graph[edge[0]][edge[1]].update(attr)

    return new_graph



def permute_features(
    features: dict, identifiers: ArrayLike, views_to_permute: List, all_views: List
) -> dict:
    """Permute the specified views for each entity (= cell line or drug)
    E.g. each cell line gets the feature vector/graph/image... of another cell line. Drawn without replacement.
    :param features: dictionary of features
    :param identifiers: array of identifiers
    :param views_to_permute: list of views to permute
    :param all_views: list of all views
    :return: permuted features
    """

    return {
        entity: {
            view: (
                features[entity][view]
                if view not in views_to_permute
                else features[other_entity][view]
            )
            for view in all_views
        }
        for entity, other_entity in zip(identifiers, np.random.permutation(identifiers))
    }


def leave_pair_out_cv(
    n_cv_splits: int,
    response: ArrayLike,
    cell_line_ids: ArrayLike,
    drug_ids: ArrayLike,
    split_validation=True,
    validation_ratio=0.1,
    random_state=42,
    dataset_name: Optional[str] = None,
) -> List[dict]:
    """
    Leave pair out cross validation. Splits data into n_cv_splits number of cross validation splits.
    :param n_cv_splits: number of cross validation splits
    :param response: response (e.g. ic50 values)
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :param split_validation: whether to split the training set into training and validation set
    :param validation_ratio: ratio of validation set (of the training set)
    :param random_state: random state
    :return: list of dicts of the cross validation sets
    """

    from drevalpy.datasets.dataset import DrugResponseDataset

    assert (
        len(response) == len(cell_line_ids) == len(drug_ids)
    ), "response, cell_line_ids and drug_ids must have the same length"

    kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)
    cv_sets = []

    for train_indices, test_indices in kf.split(response):
        if split_validation:
            # split training set into training and validation set
            train_indices, validation_indices = train_test_split(
                train_indices,
                test_size=validation_ratio,
                shuffle=True,
                random_state=random_state,
            )
        cv_fold = {
            "train": DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
                dataset_name=dataset_name,
            ),
            "test": DrugResponseDataset(
                cell_line_ids=cell_line_ids[test_indices],
                drug_ids=drug_ids[test_indices],
                response=response[test_indices],
                dataset_name=dataset_name,
            ),
        }

        if split_validation:
            cv_fold["validation"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[validation_indices],
                drug_ids=drug_ids[validation_indices],
                response=response[validation_indices],
                dataset_name=dataset_name,
            )

        cv_sets.append(cv_fold)
    return cv_sets


def leave_group_out_cv(
    group: str,
    n_cv_splits: int,
    response: ArrayLike,
    cell_line_ids: ArrayLike,
    drug_ids: ArrayLike,
    split_validation=True,
    validation_ratio=0.1,
    random_state=42,
    dataset_name: Optional[str] = None,
):
    """
    Leave group out cross validation. Splits data into n_cv_splits number of cross validation splits.
    :param group: group to leave out (cell_line or drug)
    :param n_cv_splits: number of cross validation splits
    :param random_state: random state
    :return: list of dicts of the cross validation sets
    """
    from drevalpy.datasets.dataset import DrugResponseDataset

    assert group in {
        "cell_line",
        "drug",
    }, f"group must be 'cell_line' or 'drug', but is {group}"

    if group == "cell_line":
        group_ids = cell_line_ids
    elif group == "drug":
        group_ids = drug_ids

    # shuffle, since GroupKFold does not implement this
    indices = np.arange(len(response))
    shuffled_indices = np.random.RandomState(seed=random_state).permutation(indices)
    response = response[shuffled_indices]
    cell_line_ids = cell_line_ids[shuffled_indices]
    drug_ids = drug_ids[shuffled_indices]
    group_ids = group_ids[shuffled_indices]

    gkf = GroupKFold(n_splits=n_cv_splits)
    cv_sets = []

    for train_indices, test_indices in gkf.split(response, groups=group_ids):
        cv_fold = {
            "train": DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
                dataset_name=dataset_name,
            ),
            "test": DrugResponseDataset(
                cell_line_ids=cell_line_ids[test_indices],
                drug_ids=drug_ids[test_indices],
                response=response[test_indices],
                dataset_name=dataset_name,
            ),
        }
        if split_validation:
            # split training set into training and validation set. The validation set also does contain unqiue cell lines/drugs
            unique_train_groups = np.unique(group_ids[train_indices])
            train_groups, validation_groups = train_test_split(
                unique_train_groups,
                test_size=validation_ratio,
                shuffle=True,
                random_state=random_state,
            )
            train_indices = np.where(np.isin(group_ids, train_groups))[0]
            validation_indices = np.where(np.isin(group_ids, validation_groups))[0]
            cv_fold["train"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
                dataset_name=dataset_name,
            )
            cv_fold["validation"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[validation_indices],
                drug_ids=drug_ids[validation_indices],
                response=response[validation_indices],
                dataset_name=dataset_name,
            )

        cv_sets.append(cv_fold)
    return cv_sets
