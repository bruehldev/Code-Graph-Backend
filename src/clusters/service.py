import json
import logging
from typing import List

import hdbscan
import numpy as np

from configmanager.service import ConfigManager
from data.utils import get_path_key
from database.postgresql import ClusterTable, get_cluster_table
from database.postgresql import get_data as get_all_db
from database.postgresql import (get_segment_table, get_session, init_table,
                                 table_has_entries)
from database.postgresql import update_or_create as update_or_create_db
from reduced_embeddings.service import get_reduced_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)
config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


def save_clusters(clusters: np.ndarray, index_list: List[int], dataset_name: str, model_name: str):
    logger.info(f"Save clusters db: {dataset_name} / {model_name}. Length: {len(clusters)}")
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    segment_table = get_segment_table(segment_table_name)

    init_table(cluster_table_name, cluster_table, segment_table, ClusterTable())

    for cluster, segment_id in zip(clusters, index_list):
        # if exists upda
        session = get_session()
        update_or_create_db(session, cluster_table, data_id=segment_id, cluster=int(cluster))


def get_clusters(dataset_name: str, model_name: str, start: int = 0, end: int = None):
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    if table_has_entries(cluster_table):
        clusters = get_all_db(cluster_table, start, end, True)
        return clusters
    else:
        clusters = extract_clusters(dataset_name, model_name, start, end)
        return clusters


def extract_clusters(dataset_name: str, model_name: str, start: int = 0, end: int = None):
    clusterer = hdbscan.HDBSCAN(**config.cluster_config.dict())
    reduced_embeddings_with_index = get_reduced_embeddings(dataset_name, model_name, start=0, end=None)

    # get embeddings without index
    reduced_embeddings = [embedding["reduced_embedding"] for embedding in reduced_embeddings_with_index]
    index_list = [embedding["id"] for embedding in reduced_embeddings_with_index]

    clusters = clusterer.fit_predict(reduced_embeddings)
    logger.info(f"Computed clusters: {dataset_name} / {model_name}")
    save_clusters(clusters, index_list, dataset_name, model_name)

    if end is None:
        end = len(clusters)

    clusters = [{"id": id, "cluster": clusters} for id, clusters in zip(index_list, clusters)]
    return clusters
