import json
import os
import json
import logging
from fastapi import Depends
from typing import List
from models.service import ModelService
import logging
from embeddings.service import get_reduced_embeddings
from segements.service import get_segments
from clusters.service import get_clusters
from data.few_nerd import FINE_NER_TAGS_DICT

from bertopic import BERTopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_plot(dataset_name: str, model_names: str, start: int = 0, end: int = None):
    plot_file = get_plot_file(dataset_name)
    segments = []
    if os.path.exists(plot_file):
        try:
            segments = load_plot(plot_file, start, end)
            logger.info(f"Loaded plot from file for dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading plot from file for dataset: {dataset_name}")
            logger.error(str(e))
            raise
    else:
        segments = extract_plot(dataset_name, model_names)
        segments = segments[start:end]

    logger.info(f"Retrieved plot for dataset: {dataset_name}")
    return segments


def extract_plot(dataset_name: str, model_names: str):
    # Convert index, data, embedding, and topic to JSON structure
    plot_file = get_plot_file(dataset_name)
    segments = get_segments(dataset_name)
    embeddings = get_reduced_embeddings(dataset_name, model_names)
    clusters = get_clusters(dataset_name, model_names)
    # inject embedding and cluster
    for segment, embedding, cluster in zip(segments, embeddings, clusters):
        segment["embedding"] = embedding
        segment["cluster"] = cluster
        segment["annotation"] = FINE_NER_TAGS_DICT[segment["annotation"]]
    logger.info(f"Extracted and saved plot to file for dataset: {dataset_name}")
    save_plot(segments, plot_file)
    return segments


def load_plot(plot_file, start=0, end=None):
    with open(plot_file, "r") as file:
        plot_data = json.load(file)
    return plot_data[start:end]


def save_plot(plot_data, plot_file):
    with open(plot_file, "w") as file:
        json.dump(plot_data, file)


def get_plot_file(dataset_name: str):
    plot_directory = os.path.join(env["plot_path"], dataset_name)
    os.makedirs(plot_directory, exist_ok=True)
    return os.path.join(plot_directory, f"plot_{dataset_name}.json")
