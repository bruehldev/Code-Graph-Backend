import json
import logging
import os
from typing import List

import hdbscan
import numpy as np

from configmanager.service import ConfigManager
from data.utils import get_data_file_path, get_supervised_path
from reduced_embeddings.service import get_reduced_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)
config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


def get_clusters_file(dataset_name: str, model_name: str):
    os.makedirs(get_supervised_path("clusters", dataset_name, model_name), exist_ok=True)
    return get_data_file_path(type="annotations", dataset_name=dataset_name, filename=f"clusters_{dataset_name}.json")


def save_clusters(clusters: np.ndarray, file_name: str):
    with open(file_name, "w") as f:
        json.dump(clusters, f)


def load_clusters(file_name: str) -> np.ndarray:
    with open(file_name, "r") as f:
        clusters_list = json.load(f)
        return clusters_list


def get_clusters(dataset_name: str, model_name: str, start: int = 0, end: int = None):
    logger.info(f"Getting clusters for dataset: {dataset_name}")

    clusters_file = get_clusters_file(dataset_name, model_name)

    if os.path.exists(clusters_file):
        clusters = load_clusters(clusters_file)
        logger.info(f"Loaded clusters from file for dataset: {dataset_name}")
        clusters = eval(clusters)
        # TODO Fix clusters not being a list but a string
    else:
        clusters = extract_clusters(dataset_name, model_name)

    return clusters[start:end]


def extract_clusters(dataset_name: str, model_name: str):
    clusters_file = get_clusters_file(dataset_name)
    clusterer = hdbscan.HDBSCAN(**config.cluster_config.dict())
    clusters = clusterer.fit_predict(get_reduced_embeddings(dataset_name, model_name))
    logger.info(f"Computed clusters for dataset: {dataset_name} with length: {len(clusters)}")
    # convert the clusters to a JSON serializable format
    clusters = [int(c) for c in clusterer.labels_]
    # serialize the clusters to JSON
    json_clusters = json.dumps(clusters)
    save_clusters(json_clusters, clusters_file)
    logger.info(f"Computed and saved clusters for dataset: {dataset_name}")
    clusters = list(clusters)
    return clusters
