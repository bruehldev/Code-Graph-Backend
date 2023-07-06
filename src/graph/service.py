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
from embeddings.service import get_embeddings

from bertopic import BERTopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_graph(dataset_name: str, model: BERTopic = Depends(load_model), embeddings: list = Depends(get_embeddings), data: list = Depends(get_data)):
    positions_file = get_positions_file(dataset_name)

    if os.path.exists(positions_file):
        try:
            positions_dict = load_positions(positions_file)
            logger.info(f"Loaded positions from file for dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading positions from file for dataset: {dataset_name}")
            logger.error(str(e))
            raise
    else:
        # Convert index, data, embedding, and topic to JSON structure
        positions_dict = {}
        for index, (embedding, doc) in enumerate(zip(embeddings[:20], data[:20])):
            position = {"data": doc, "position": embedding, "topic_index": np.array(model.transform([str(embedding)]))[0].tolist()}
            positions_dict[str(index)] = position

        save_positions(positions_dict, positions_file)
        logger.info(f"Saved positions to file for dataset: {dataset_name}")

    positions = list(zip(data, positions_dict.values()))

    logger.info(f"Retrieved positions for dataset: {dataset_name}")
    return positions


def load_positions(positions_file):
    with open(positions_file, "r") as file:
        positions_data = json.load(file)
    return positions_data


def save_positions(positions_data, positions_file):
    with open(positions_file, "w") as file:
        json.dump(positions_data, file)


def get_positions_file(dataset_name: str):
    positions_directory = os.path.join(env["positions_path"], dataset_name)
    os.makedirs(positions_directory, exist_ok=True)
    return os.path.join(positions_directory, f"positions_{dataset_name}.json")
