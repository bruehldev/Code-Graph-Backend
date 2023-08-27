import os
import json
import logging
import pickle
import logging
import numpy as np
import umap

from typing import List
from data.service import get_data
from segments.service import get_segments
from models.service import ModelService
from data.utils import get_model_file_path, get_supervised_path, get_path_key
from configmanager.service import ConfigManager
from database.postgresql import get_segment_table, get as get_in_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)

config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


### Embedding functions ###
def get_embeddings(dataset_name: str, model_name: str, start=0, end=None, with_id: bool = False):
    global embeddings_2d_bert
    embeddings_file = get_embeddings_file(dataset_name, model_name)
    if os.path.exists(embeddings_file):
        embeddings_2d_bert = load_embeddings(dataset_name, model_name, with_id)[start:end]
    else:
        embeddings_2d_bert = extract_embeddings(dataset_name=dataset_name, model_name=model_name, start=start, end=end, id=None, return_with_id=with_id)

    if isinstance(embeddings_2d_bert, np.ndarray):
        embeddings_2d_bert = embeddings_2d_bert.tolist()

    logger.info(f"Returning len {len(embeddings_2d_bert)} start {start} end {end}")
    return embeddings_2d_bert


def save_embeddings(embeddings: np.ndarray, dataset_name: str, model_name: str, start=0, end=None):
    embeddings_file = get_embeddings_file(dataset_name, model_name)
    num_embeddings = len(embeddings)  # Get the number of embeddings from the list

    logger.info(f"Save embeddings as pickle: {dataset_name} / {model_name}. Length: {num_embeddings}")

    if end is None:
        end = num_embeddings

    with open(embeddings_file, "wb") as f:
        # Only save embeddings within the specified range
        pickle.dump((num_embeddings, embeddings[start:end]), f)


def load_embeddings(dataset_name: str, model_name: str, with_index: bool = False) -> List:
    logger.info(f"Loaded embeddings pickle {dataset_name} / {model_name} with index: {with_index}")
    embeddings_file = get_embeddings_file(dataset_name, model_name)
    with open(embeddings_file, "rb") as f:
        num_embeddings, indexed_embeddings = pickle.load(f)
        # sort by index
        indexed_embeddings = sorted(indexed_embeddings, key=lambda x: x[0])
    if with_index:
        return indexed_embeddings
    else:
        embeddings = [embedding for index, embedding in indexed_embeddings]
        return embeddings


def delete_embeddings(dataset_name: str, model_name: str) -> bool:
    embeddings_file = get_embeddings_file(dataset_name, model_name)
    if os.path.exists(embeddings_file):
        logger.info(f"Deleting embeddings.  dataset: {dataset_name} modelname: {model_name}")
        os.remove(embeddings_file)
        return True
    else:
        logger.info(f"Embeddings file does not exist.  dataset: {dataset_name} modelname: {model_name}")
        return False


def create_embedding(id: int, embedding: list, dataset_name: str, model_name: str) -> List:
    loaded_embeddings = load_embeddings(dataset_name, model_name, with_index=True)

    # Get the index for the new embedding
    last_index = loaded_embeddings[-1][0] if loaded_embeddings else 0
    new_index = last_index + 1
    logger.info(f"Creating embedding.  dataset: {dataset_name} modelname: {model_name} index: {new_index}")

    # Append the new embedding with its index
    if id is not None:
        new_index = id
    loaded_embeddings.append([new_index, embedding])
    save_embeddings(embeddings=loaded_embeddings, dataset_name=dataset_name, model_name=model_name)

    return [new_index, embedding]


def read_embedding(index: int, dataset_name: str, model_name: str) -> int and np.ndarray:
    logger.info(f"reading embedding.  dataset: {dataset_name} modelname: {model_name} index: {index}")

    loaded_embeddings = load_embeddings(dataset_name, model_name, with_index=True)

    for loaded_index, embedding in loaded_embeddings:
        if loaded_index == index:
            return loaded_index, embedding
    return None, None


def update_embedding(index: int, new_embedding: np.ndarray, dataset_name: str, model_name: str) -> List:
    logger.info(f"Updating embedding.  dataset: {dataset_name} modelname: {model_name} index: {index}")
    loaded_embeddings = load_embeddings(dataset_name, model_name, with_index=True)

    for i, (loaded_index, embedding) in enumerate(loaded_embeddings):
        if loaded_index == index:
            loaded_embeddings[i] = [loaded_index, new_embedding]
            save_embeddings(loaded_embeddings, dataset_name, model_name)
            return [loaded_index, new_embedding]


def delete_embedding(index: int, dataset_name: str, model_name: str) -> int:
    logger.info(f"Deleting embedding.  dataset: {dataset_name} modelname: {model_name} index: {index}")
    loaded_embeddings = load_embeddings(dataset_name, model_name, with_index=True)

    for i, (loaded_index, embedding) in enumerate(loaded_embeddings):
        if loaded_index == index:
            loaded_embeddings.pop(i)
            save_embeddings(loaded_embeddings, dataset_name, model_name)
            return loaded_index
    return None


def get_embeddings_file(dataset_name: str, model_name: str):
    embeddings_directory = get_supervised_path("embeddings", dataset_name, model_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return get_model_file_path(type="embeddings", dataset_name=dataset_name, model_name=model_name, filename=f"embeddings_{dataset_name}.pkl")


def extract_embeddings(dataset_name, model_name, start=0, end=None, id=None, return_with_id=None) -> List:
    model_service = ModelService(dataset_name, model_name)
    logger.info(f"Extract embeddings: {dataset_name} / {model_name} start: {start} end: {end}, id: {id}")
    segments = []

    if id is not None:
        table_name = get_path_key("data", dataset_name)
        segment_table = get_segment_table(table_name)
        segments.append(get_in_db(segment_table, id))
    elif start is not None and end is not None:
        segments.extend(get_segments(dataset_name, start, end))
    else:
        segments.extend(get_segments(dataset_name))

    if dataset_name == "few_nerd":
        embeddings = model_service.process_data(segments)
        embeddings_2d_bert = embeddings
    elif dataset_name == "fetch_20newsgroups":
        embeddings = model_service.model(get_data(dataset_name, start, end))
        umap_model = umap.UMAP(**config.embedding_config.dict())
        embeddings_2d_bert = umap_model.fit_transform(embeddings)

    logger.info(f"Computed embeddings: {dataset_name} / {model_name}")
    embeddings_file = get_embeddings_file(dataset_name, model_name)

    if os.path.exists(embeddings_file):
        # update or create for each new embedding
        for embedding in embeddings_2d_bert:
            embedding_id: int = embedding[0]
            embedding: list = embedding[1]

            # Todo Fix fileoperation with db or delta load. We can have huge perfomance issues on large amount of extrations
            if read_embedding(embedding_id, dataset_name, model_name) is not None:
                update_embedding(embedding_id, embedding, dataset_name, model_name)
            else:
                create_embedding(embedding_id, embedding, dataset_name, model_name)
    else:
        save_embeddings(embeddings_2d_bert, dataset_name, model_name, start, end)

    if id is not None:
        return [read_embedding(id, dataset_name, model_name)]

    return load_embeddings(dataset_name, model_name, return_with_id)[start:end]
