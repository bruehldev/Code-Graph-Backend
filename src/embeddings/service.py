import json
import os
import json
import logging
from typing import List
from data.service import get_data, get_segments
from models.service import ModelService
import logging
import numpy as np
import umap


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
    with open(file_name, "w") as f:
        json.dump(embeddings, f)


def load_embeddings(file_name: str) -> np.ndarray:
    with open(file_name, "r") as f:
        embeddings_list = json.load(f)
        return np.array(embeddings_list)


def get_embeddings_file(dataset_name: str):
    embeddings_directory = os.path.join(env["embeddings_path"], dataset_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return os.path.join(embeddings_directory, f"embeddings_{dataset_name}.json")


def save_reduced_embeddings(reduced_embeddings: np.ndarray, file_name: str):
    with open(file_name, "w") as f:
        json.dump(reduced_embeddings.tolist(), f)


def load_reduced_embeddings(file_name: str) -> np.ndarray:
    with open(file_name, "r") as f:
        reduced_embeddings_list = json.load(f)
        return np.array(reduced_embeddings_list)


def get_reduced_embeddings_file(dataset_name: str):
    embeddings_directory = os.path.join(env["embeddings_path"], dataset_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return os.path.join(embeddings_directory, f"reduced_embeddings_{dataset_name}.json")


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


def get_embeddings(dataset_name: str, model_name: str):
    global embeddings_2d_bert
    embeddings_file = get_embeddings_file(dataset_name)

    if os.path.exists(embeddings_file):
        embeddings_2d_bert = load_embeddings(embeddings_file)
        logger.info(f"Loaded embeddings from file for dataset: {dataset_name}")
    else:
        embeddings_2d_bert = extract_embeddings(dataset_name, model_name)
        save_embeddings(embeddings_2d_bert, embeddings_file)

        logger.info(f"Computed and saved embeddings for dataset: {dataset_name}")

    if isinstance(embeddings_2d_bert, np.ndarray):
        embeddings_2d_bert = embeddings_2d_bert.tolist()

    return len(embeddings_2d_bert)


def get_reduced_embeddings(dataset_name: str, model_name: str):
    embeddings_file = get_reduced_embeddings_file(dataset_name)
    embeddings_reduced = None
    if os.path.exists(embeddings_file):
        embeddings_reduced = load_reduced_embeddings(embeddings_file)
        logger.info(f"Loaded embeddings from file for dataset: {dataset_name}")
    else:
        embeddings = get_embeddings(dataset_name, model_name)
        umap_model = umap.UMAP(**config.embedding_config.dict())
        embeddings_reduced = umap_model.fit_transform(embeddings)
        embeddings_file = get_reduced_embeddings_file(dataset_name)
        save_reduced_embeddings(embeddings_reduced, embeddings_file)

        logger.info(f"Computed and saved embeddings for dataset: {dataset_name}")

    return embeddings_reduced
