import json
import os
import json
import logging
from fastapi import Depends
from typing import List
from data.service import get_data
from models.service import ModelService
import logging
import numpy as np
from embeddings.service import get_reduced_embeddings, get_segments
from clusters.service import get_clusters

from bertopic import BERTopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_plot(dataset_name: str, model_names: str):
    plot_file = get_plot_file(dataset_name)
    segments = []
    if os.path.exists(plot_file):
        try:
            segments = load_plot(plot_file)
            logger.info(f"Loaded plot from file for dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading plot from file for dataset: {dataset_name}")
            logger.error(str(e))
            raise
    else:
        # Convert index, data, embedding, and topic to JSON structure
        segments = get_segments(dataset_name, model_names)[:500]
        embeddings = get_reduced_embeddings(dataset_name, model_names).tolist()[:500]
        clusters = get_clusters(dataset_name, model_names)[:500]

        # inject embedding and cluster
        for segment, embedding, cluster in zip(segments, embeddings, clusters):
            segment["embedding"] = embedding
            segment["cluster"] = cluster

        # logger.info(f"Saved plot to file for dataset: {dataset_name}: {segments}")
        save_plot(segments, plot_file)

    logger.info(f"Retrieved plot for dataset: {dataset_name}")
    return segments


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
