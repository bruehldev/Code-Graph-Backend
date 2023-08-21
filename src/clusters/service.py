import json
import logging
from typing import List
from embeddings.service import get_embeddings
from database.postgresql import ClustersTable, table_has_entries, get_data as get_data_db, init_table, create
from data.utils import get_path_key

import logging
import numpy as np
import hdbscan


from configmanager.service import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)
config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


def save_clusters(clusters: np.ndarray, dataset_name: str, model_name: str):
    logger.info(f"Save clusters db: {dataset_name} / {model_name}. Length: {len(clusters)}")
    path_key = get_path_key(type="clusters", dataset_name=dataset_name, model_name=model_name)
    init_table(path_key, ClustersTable)

    for cluster in clusters:
        create(path_key, ClustersTable, cluster=int(cluster))


def get_clusters(dataset_name: str, model_name: str, start: int = 0, end: int = None):
    path_key = get_path_key(type="clusters", dataset_name=dataset_name, model_name=model_name)
    if table_has_entries(path_key, ClustersTable):
        clusters = get_data_db(path_key, start, end, ClustersTable)
        return clusters
    else:
        clusters = extract_clusters(dataset_name, model_name)
        save_clusters(clusters, dataset_name, model_name)
        return clusters[start:end]


def extract_clusters(dataset_name: str, model_name: str):
    clusterer = hdbscan.HDBSCAN(**config.cluster_config.dict())
    clusters = clusterer.fit_predict(get_embeddings(dataset_name, model_name))
    logger.info(f"Computed clusters: {dataset_name} / {model_name}")
    save_clusters(clusters, dataset_name, model_name)
    return clusters
