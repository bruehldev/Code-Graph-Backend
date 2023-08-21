import os
import json
import logging
import pickle
import logging
import numpy as np
import umap

from typing import List
from data.service import get_data
from segements.service import get_segments
from data.utils import get_model_file_path, get_supervised_path, get_path_key
from configmanager.service import ConfigManager
from database.postgresql import (
    SessionLocal,
    ReducedEmbeddingsTable,
    init_table,
    create,
    get_data as get_data_db,
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
    path_key = get_path_key(type="reduced_embedding", dataset_name=dataset_name, model_name=model_name)
    embeddings_reduced = []

    if table_has_entries(path_key, ReducedEmbeddingsTable):
        embeddings_reduced = get_data_db(path_key, start, end, ReducedEmbeddingsTable)
        return embeddings_reduced
        # return [row.__dict__ for row in data]
        # embeddings_reduced = load_reduced_embeddings(dataset_name, model_name, start, end)
    else:
        embeddings_reduced = extract_embeddings_reduced(dataset_name, model_name)

        if isinstance(embeddings_reduced, np.ndarray):
            embeddings_reduced = embeddings_reduced.tolist()
        return embeddings_reduced[start:end]


def save_reduced_embeddings(reduced_embeddings: np.ndarray, dataset_name: str, model_name: str):
    logger.info(f"Save reduced embeddings db: {dataset_name} / {model_name}. Length: {len(reduced_embeddings)}")
    path_key = get_path_key(type="reduced_embedding", dataset_name=dataset_name, model_name=model_name)
    init_table(path_key, ReducedEmbeddingsTable)

    for embedding in reduced_embeddings:
        create(path_key, ReducedEmbeddingsTable, reduced_embeddings=embedding.tolist())


def get_reduced_embeddings_file(dataset_name: str, model_name: str):
    embeddings_directory = get_supervised_path("embeddings", dataset_name, model_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return get_model_file_path(type="embeddings", dataset_name=dataset_name, model_name=model_name, filename=f"reduced_embeddings_{dataset_name}.pkl")


def extract_embeddings_reduced(dataset_name, model_name):
    embeddings = np.array(get_embeddings(dataset_name, model_name))
    umap_model = umap.UMAP(**config.embedding_config.dict())

    # Check the shape of the embeddings array
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)  # Reshape to a column vector

    embeddings_reduced = umap_model.fit_transform(embeddings)

    save_reduced_embeddings(embeddings_reduced, dataset_name, model_name)
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
