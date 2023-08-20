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
from models.service import ModelService
from data.utils import get_model_file_path, get_supervised_path, get_path_key
from configmanager.service import ConfigManager
from database.postgresql import (
    SessionLocal,
    SegmentsTable,
    ReducedEmbeddingsTable,
    init_table,
    create,
    ReducedEmbeddingsTable,
    get_data as get_data_db,
    table_has_entries,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)

config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


### Embedding functions ###
def get_embeddings(dataset_name: str, model_name: str, start=0, end=None):
    global embeddings_2d_bert
    embeddings_file = get_embeddings_file(dataset_name, model_name)

    if os.path.exists(embeddings_file):
        embeddings_2d_bert = load_embeddings(dataset_name, model_name)
    else:
        embeddings_2d_bert = extract_embeddings(dataset_name, model_name)
        save_embeddings(embeddings_2d_bert, embeddings_file)

    if isinstance(embeddings_2d_bert, np.ndarray):
        embeddings_2d_bert = embeddings_2d_bert.tolist()

    logger.info(f"Returning len {len(embeddings_2d_bert[start:end])} start {start} end {end}")
    return embeddings_2d_bert[start:end]


def save_embeddings(embeddings: np.ndarray, dataset_name: str, model_name: str, index_provided: bool = False):
    embeddings_file = get_embeddings_file(dataset_name, model_name)
    num_embeddings = len(embeddings)  # Get the number of embeddings from the list

    if not index_provided:
        embeddings = [(index, embedding) for index, embedding in enumerate(embeddings)]

    logger.info(f"Save embeddings in db: {dataset_name} / {model_name}. Length: {num_embeddings}")

    with open(embeddings_file, "wb") as f:
        pickle.dump((num_embeddings, embeddings), f)


def load_embeddings(dataset_name: str, model_name: str, with_index: bool = False) -> List:
    logger.info(f"Loaded embeddings pickle {dataset_name} / {model_name} with index: {with_index}")
    embeddings_file = get_embeddings_file(dataset_name, model_name)

    with open(embeddings_file, "rb") as f:
        num_embeddings, indexed_embeddings = pickle.load(f)

    if with_index:
        return indexed_embeddings
    else:
        embeddings = [embedding for index, embedding in indexed_embeddings]
        return embeddings


def create_embedding(embedding: list, dataset_name: str, model_name: str) -> List:
    logger.info(f"Creating embedding.  dataset: {dataset_name} modelname: {model_name} index: {new_index}")
    loaded_embeddings = load_embeddings(dataset_name, model_name, with_index=True)

    # Get the index for the new embedding
    last_index = loaded_embeddings[-1][0] if loaded_embeddings else -1
    new_index = last_index + 1

    # Append the new embedding with its index
    loaded_embeddings.append([new_index, embedding])
    save_embeddings(embeddings=loaded_embeddings, dataset_name=dataset_name, model_name=model_name, index_provided=True)


def read_embedding(index: int, dataset_name: str, model_name: str) -> np.ndarray:
    logger.info(f"reading embedding.  dataset: {dataset_name} modelname: {model_name} index: {index}")

    loaded_embeddings = load_embeddings(dataset_name, model_name, with_index=True)

    for loaded_index, embedding in loaded_embeddings:
        if loaded_index == index:
            return embedding
    return None


def update_embedding(index: int, new_embedding: np.ndarray, dataset_name: str, model_name: str):
    logger.info(f"Updating embedding.  dataset: {dataset_name} modelname: {model_name} index: {index}")
    loaded_embeddings = load_embeddings(dataset_name, model_name, with_index=True)

    for i, (loaded_index, embedding) in enumerate(loaded_embeddings):
        if loaded_index == index:
            loaded_embeddings[i] = [loaded_index, new_embedding]
            save_embeddings(loaded_embeddings, dataset_name, model_name, index_provided=True)
            return


def delete_embedding(index: int, dataset_name: str, model_name: str):
    logger.info(f"Deleting embedding.  dataset: {dataset_name} modelname: {model_name} index: {index}")
    loaded_embeddings = load_embeddings(dataset_name, model_name, with_index=True)

    for i, (loaded_index, embedding) in enumerate(loaded_embeddings):
        if loaded_index == index:
            loaded_embeddings.pop(i)
            save_embeddings(loaded_embeddings, dataset_name, model_name, index_provided=True)
            return


def get_embeddings_file(dataset_name: str, model_name: str):
    embeddings_directory = get_supervised_path("embeddings", dataset_name, model_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return get_model_file_path(type="embeddings", dataset_name=dataset_name, model_name=model_name, filename=f"embeddings_{dataset_name}.pkl")


def extract_embeddings(dataset_name, model_name):
    model_service = ModelService(dataset_name, model_name)
    logger.info("Extracting embeddings for documents")

    if dataset_name == "few_nerd":
        embeddings = model_service.process_data(get_segments(dataset_name))
        embeddings_2d_bert = embeddings
    elif dataset_name == "fetch_20newsgroups":
        embeddings = model_service.model(get_data(dataset_name))
        umap_model = umap.UMAP(**config.embedding_config.dict())
        embeddings_2d_bert = umap_model.fit_transform(embeddings)

    logger.info(f"Computed embeddings: {dataset_name} / {model_name}")
    save_embeddings(embeddings_2d_bert, dataset_name, model_name)
