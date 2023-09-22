import json
import logging
import os
import pickle
from typing import List

import numpy as np
import umap
from tqdm import tqdm

from configmanager.service import ConfigManager
from data.service import get_data
from data.utils import get_model_file_path, get_path_key, get_supervised_path
from database.postgresql import EmbeddingTable, batch_insert
from database.postgresql import get as get_in_db
from database.postgresql import get_data as get_all_db
from database.postgresql import (get_embedding_table, get_segment_table,
                                 get_session, init_table, table_has_entries,
                                 update_or_create)
from models.service import ModelService
from segments.service import get_segments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)

config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


### Embedding functions ###
def get_embeddings(dataset_name: str, model_name: str, start=0, end=None):
    segment_table_name = get_path_key("segments", dataset_name)
    embedding_table_name = get_path_key("embeddings", dataset_name, model_name)
    embeddings_table = get_embedding_table(embedding_table_name, segment_table_name)

    embeddings = []

    if table_has_entries(embeddings_table):
        embeddings = get_all_db(embeddings_table, start, end, True)
        embeddings = [{"id": embedding["id"], "embedding": pickle.loads(embedding["embedding"])} for embedding in embeddings]
    else:
        embeddings = extract_embeddings(dataset_name, model_name, start, end)

    return embeddings


def save_embeddings(embeddings: np.ndarray, dataset_name: str, model_name: str):
    segments_table_name = get_path_key("segments", dataset_name)
    segments_table = get_segment_table(segments_table_name)
    embeddings_table_name = get_path_key("embeddings", dataset_name, model_name)
    embeddings_table = get_embedding_table(embeddings_table_name, segments_table_name)
    init_table(embeddings_table_name, embeddings_table, segments_table, EmbeddingTable())

    session = get_session()
    total_entries = len(embeddings)

    # transform embeddings to list of dicts

    with tqdm(total=total_entries, desc=f"Saving {dataset_name} / {model_name}") as pbar:
        for embedding in embeddings:
            # embedding example: [1, [-1.018982291221618...
            update_or_create(session, embeddings_table, data_id=embedding[0], embedding=pickle.dumps(embedding[1]))
            pbar.update(1)

    session.commit()


def extract_embeddings(dataset_name, model_name, start=0, end=None, id=None) -> List:
    model_service = ModelService(dataset_name, model_name)
    logger.info(f"Extract embeddings: {dataset_name} / {model_name} start: {start} end: {end}, id: {id}")
    segments = []

    if id is not None:
        table_name = get_path_key("segments", dataset_name)
        segment_table = get_segment_table(table_name)
        segments.append(get_in_db(segment_table, id))
    elif start is not None and end is not None:
        segments.extend(get_segments(dataset_name, start, end))
    else:
        segments.extend(get_segments(dataset_name))

    if dataset_name == "few_nerd":
        embeddings = model_service.process_data(segments)
        id_embedding_array = embeddings
    elif dataset_name == "fetch_20newsgroups":
        embeddings = model_service.model(get_data(dataset_name, start, end))
        umap_model = umap.UMAP(**config.reduction_config.dict())
        id_embedding_array = umap_model.fit_transform(embeddings)

    logger.info(f"Computed embeddings: {dataset_name} / {model_name}")
    save_embeddings(id_embedding_array, dataset_name, model_name)
    # transform to dict
    embeddings = [{"id": embedding[0], "embedding": embedding[1]} for embedding in id_embedding_array]
    return embeddings
