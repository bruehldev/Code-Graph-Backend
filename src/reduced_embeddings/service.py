import os
import json
import logging
import pickle
import logging
import numpy as np
import umap

from typing import List
from data.utils import get_model_file_path, get_supervised_path, get_path_key
from configmanager.service import ConfigManager
from database.postgresql import (
    get_reduced_embedding_table,
    get_segment_table,
    init_table,
    create as create_in_db,
    get_data as get_all_db,
    table_has_entries,
)
from embeddings.service import get_embeddings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)

config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


# Keeping file operations for now
### Reduced Embedding functions ###
def get_reduced_embeddings(dataset_name: str, model_name: str, start=0, end=None):
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)

    embeddings_reduced = []

    if table_has_entries(reduced_embeddings_table):
        embeddings_reduced = get_all_db(reduced_embeddings_table, start, end, True)
        return embeddings_reduced
        # return [row.__dict__ for row in data]
        # embeddings_reduced = load_reduced_embeddings(dataset_name, model_name, start, end)
    else:
        embeddings_reduced = extract_embeddings_reduced(dataset_name, model_name, start, end)

        if isinstance(embeddings_reduced, np.ndarray):
            embeddings_reduced = embeddings_reduced.tolist()
        return embeddings_reduced[start:end]


def save_reduced_embeddings(reduced_embeddings: np.ndarray, index_list: List[int], dataset_name: str, model_name: str):
    logger.info(f"Save reduced embeddings db: {dataset_name} / {model_name}. Length: {len(reduced_embeddings)}")
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(segment_table_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)

    init_table(reduced_embedding_table_name, reduced_embeddings_table, segment_table)

    for embedding, segment_id in zip(reduced_embeddings, index_list):
        create_in_db(reduced_embeddings_table, reduced_embeddings=embedding.tolist(), segment_id=segment_id)


def get_reduced_embeddings_file(dataset_name: str, model_name: str):
    embeddings_directory = get_supervised_path("embeddings", dataset_name, model_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return get_model_file_path(type="embeddings", dataset_name=dataset_name, model_name=model_name, filename=f"reduced_embeddings_{dataset_name}.pkl")


def extract_embeddings_reduced(dataset_name, model_name, start=0, end=None):
    # embeddings with index
    embeddings_with_index = get_embeddings(dataset_name, model_name, start, end, with_index=True)
    # get embeddings without index
    embeddings = [embedding for index, embedding in embeddings_with_index]
    index_list = [index for index, embedding in embeddings_with_index]
    reduced_embeddings = np.array(embeddings)
    umap_model = umap.UMAP(**config.embedding_config.dict())

    # Check the shape of the embeddings array
    if reduced_embeddings.ndim == 1:
        reduced_embeddings = reduced_embeddings.reshape(-1, 1)  # Reshape to a column vector

    embeddings_reduced = umap_model.fit_transform(reduced_embeddings)

    save_reduced_embeddings(embeddings_reduced, index_list, dataset_name, model_name)
    logger.info(f"Computed reduced embeddings: {dataset_name} / {model_name}")
    return embeddings_reduced


### Save load reduced embeddings from pickle file and not db ###
def save_reduced_embeddings_pickle(reduced_embeddings: np.ndarray, dataset_name: str, model_name: str):
    embeddings_file = get_reduced_embeddings_file(dataset_name, model_name)
    with open(embeddings_file, "wb") as f:
        pickle.dump(reduced_embeddings, f)


def get_reduced_embeddings_pickle(dataset_name, model_name):
    embeddings_file = get_reduced_embeddings_file(dataset_name, model_name)

    with open(embeddings_file, "rb") as f:
        reduced_embeddings = pickle.load(f)
        return reduced_embeddings
