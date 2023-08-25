import json
import logging
from typing import List
from embeddings.service import get_embeddings
from database.postgresql import get_cluster_table, table_has_entries, get_data as get_all_db, init_table, create as create_in_db
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
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    init_table(cluster_table_name, cluster_table)

    for cluster in clusters:
        create_in_db(cluster_table, cluster=int(cluster))


def get_clusters(dataset_name: str, model_name: str, start: int = 0, end: int = None):
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    if table_has_entries(cluster_table):
        clusters = get_all_db(cluster_table, start, end, True)
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
