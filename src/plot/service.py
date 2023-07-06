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


def get_plot(dataset_name: str, model: BERTopic = Depends(load_model), embeddings: list = Depends(get_embeddings), data: list = Depends(get_data)):
    plot_file = get_plot_file(dataset_name)

    if os.path.exists(plot_file):
        try:
            plot_dict = load_plot(plot_file)
            logger.info(f"Loaded plot from file for dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading plot from file for dataset: {dataset_name}")
            logger.error(str(e))
            raise
    else:
        # Convert index, data, embedding, and topic to JSON structure
        plot_dict = {}
        for index, (embedding, doc) in enumerate(zip(embeddings[:20], data[:20])):
            plot = {"data": doc, "plot": embedding, "topic_index": np.array(model.transform([str(embedding)]))[0].tolist()}
            plot_dict[str(index)] = plot

        save_plot(plot_dict, plot_file)
        logger.info(f"Saved plot to file for dataset: {dataset_name}")

    plot = list(zip(data, plot_dict.values()))

    logger.info(f"Retrieved plot for dataset: {dataset_name}")
    return plot


def load_plot(plot_file):
    with open(plot_file, "r") as file:
        plot_data = json.load(file)
    return plot_data


def save_plot(plot_data, plot_file):
    with open(plot_file, "w") as file:
        json.dump(plot_data, file)


def get_plot_file(dataset_name: str):
    plot_directory = os.path.join(env["plot_path"], dataset_name)
    os.makedirs(plot_directory, exist_ok=True)
    return os.path.join(plot_directory, f"plot_{dataset_name}.json")
