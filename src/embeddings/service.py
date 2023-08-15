import os
import json
import logging
import pickle
from typing import List
from data.service import get_data, get_segments
from models.service import ModelService
import logging
import numpy as np
import umap
from data.utils import get_path_key, get_file_path, get_root_path, get_supervised_path


from configmanager.service import ConfigManager
from bertopic import BERTopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)

config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


def save_embeddings(embeddings: np.ndarray, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings(file_name: str) -> np.ndarray:
    with open(file_name, "rb") as f:
        embeddings = pickle.load(f)
        return embeddings


def get_embeddings_file(dataset_name: str):
    embeddings_directory = get_supervised_path("embeddings", dataset_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return get_file_path("embeddings", dataset_name, f"embeddings_{dataset_name}.pkl")


def save_reduced_embeddings(reduced_embeddings: np.ndarray, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(reduced_embeddings, f)


def load_reduced_embeddings(file_name: str) -> np.ndarray:
    with open(file_name, "rb") as f:
        reduced_embeddings = pickle.load(f)
        return reduced_embeddings


def get_reduced_embeddings_file(dataset_name: str):
    embeddings_directory = get_supervised_path("embeddings", dataset_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return get_file_path("embeddings", dataset_name, f"reduced_embeddings_{dataset_name}.pkl")


def extract_embeddings(dataset_name, model_name):
    model_service = ModelService(dataset_name, model_name)
    logger.info("Extracting embeddings for documents")
    if dataset_name == "few_nerd":
        embeddings = model_service.process_data(get_segments(dataset_name))
        return embeddings
    elif dataset_name == "fetch_20newsgroups":
        embeddings = model_service.model(get_data(dataset_name))
        umap_model = umap.UMAP(**config.embedding_config.dict())
        return umap_model.fit_transform(embeddings)


def get_embeddings(dataset_name: str, model_name: str, start=0, end=None):
    global embeddings_2d_bert
    embeddings_file = get_embeddings_file(dataset_name)

    if os.path.exists(embeddings_file):
        embeddings_2d_bert = load_embeddings(embeddings_file)
        logger.info(f"Loaded embeddings from pickle file for dataset: {dataset_name}")
    else:
        embeddings_2d_bert = extract_embeddings(dataset_name, model_name)
        save_embeddings(embeddings_2d_bert, embeddings_file)
        logger.info(f"Computed and saved embeddings for dataset: {dataset_name}")

    if isinstance(embeddings_2d_bert, np.ndarray):
        embeddings_2d_bert = embeddings_2d_bert.tolist()

    print("embeddings_2d_bert ", len(embeddings_2d_bert))
    return embeddings_2d_bert[start:end]


def get_reduced_embeddings(dataset_name: str, model_name: str, start=0, end=None):
    embeddings_file = get_reduced_embeddings_file(dataset_name)
    embeddings_reduced = []

    if os.path.exists(embeddings_file):
        embeddings_reduced = load_reduced_embeddings(embeddings_file)
        logger.info(f"Loaded embeddings from pickle file for dataset: {dataset_name}")
    else:
        embeddings_reduced = extract_embeddings_reduced(dataset_name, model_name)

    print("embeddings_reduced ", len(embeddings_reduced))
    if isinstance(embeddings_reduced, np.ndarray):
        embeddings_reduced = embeddings_reduced.tolist()

    return embeddings_reduced[start:end]


def extract_embeddings_reduced(dataset_name, model_name):
    embeddings = np.array(get_embeddings(dataset_name, model_name))
    # embeddings = get_embeddings(dataset_name, model_name)
    umap_model = umap.UMAP(**config.embedding_config.dict())

    # Check the shape of the embeddings array
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)  # Reshape to a column vector

    embeddings_reduced = umap_model.fit_transform(embeddings)
    embeddings_file = get_reduced_embeddings_file(dataset_name)
    save_reduced_embeddings(embeddings_reduced, embeddings_file)
    logger.info(f"Computed and saved embeddings for dataset: {dataset_name}")
    return embeddings_reduced
