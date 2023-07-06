import json
import os
import json
import logging
from fastapi import Depends
from typing import List
from data.service import get_data
from models.service import load_model
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
        json.dump(embeddings.tolist(), f)


def load_embeddings(file_name: str) -> np.ndarray:
    with open(file_name, "r") as f:
        embeddings_list = json.load(f)
        return np.array(embeddings_list)


def get_embeddings_file(dataset_name: str):
    embeddings_directory = os.path.join(env["embeddings_path"], dataset_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return os.path.join(embeddings_directory, f"embeddings_{dataset_name}.json")


def extract_embeddings(model, data):
    logger.info("Extracting embeddings for documents")
    embeddings = model._extract_embeddings(data)
    umap_model = umap.UMAP(**config.embedding_config.dict())
    return umap_model.fit_transform(embeddings)


def get_embeddings(dataset_name: str, model: BERTopic = Depends(load_model), data: list = Depends(get_data)):
    global embeddings_2d_bert
    embeddings_file = get_embeddings_file(dataset_name)

    if os.path.exists(embeddings_file):
        embeddings_2d_bert = load_embeddings(embeddings_file)
        logger.info(f"Loaded embeddings from file for dataset: {dataset_name}")
    else:
        embeddings_2d_bert = extract_embeddings(model, data)
        save_embeddings(embeddings_2d_bert, embeddings_file)
        logger.info(f"Computed and saved embeddings for dataset: {dataset_name}")

    return embeddings_2d_bert.tolist()
