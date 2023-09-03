import json
import os
import json
import logging
import csv

from reduced_embeddings.service import get_reduced_embeddings
from segments.service import get_segments
from clusters.service import get_clusters
from data.few_nerd import FINE_NER_TAGS_DICT
from data.utils import get_supervised_path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_plot(dataset_name: str, model_name: str, start: int = 0, end: int = None):
    plot_file = get_plot_file(dataset_name, model_name, suffix="json")
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
        segments = extract_plot(dataset_name, model_name)
        segments = segments[start:end]

    logger.info(f"Retrieved plot for dataset: {dataset_name}")
    return segments


def extract_plot(dataset_name: str, model_name: str):
    # Convert index, data, embedding, and topic to JSON structure
    json_plot_file = get_plot_file(dataset_name, model_name, suffix="json")
    csv_plot_file = get_plot_file(dataset_name, model_name, suffix="csv")
    segments = get_segments(dataset_name)
    embeddings = get_reduced_embeddings(dataset_name, model_name)
    clusters = get_clusters(dataset_name, model_name)

    for segment, embedding, cluster in zip(segments, embeddings, clusters):
        # Check if ids are equal
        assert segment["id"] == embedding["id"] == cluster["id"]

        segment["reduced_embedding"] = embedding["reduced_embedding"]
        segment["cluster"] = cluster["cluster"]
        segment["annotation"] = segment["annotation"]

    logger.info(f"Extracted and saved plot to file for dataset: {dataset_name}")
    save_plot_json(segments, json_plot_file)
    save_plot_csv(segments, csv_plot_file)
    return segments


def save_plot_csv(plot_data, plot_file):
    with open(plot_file, "w", newline="") as file:
        csv_writer = csv.writer(file)

        # Write header row
        header = ["id", "sentence", "reduced_embedding", "cluster", "annotation"]
        csv_writer.writerow(header)

        # Write plot data rows
        for segment in plot_data:
            index = segment["id"]
            data = segment["sentence"]
            embedding = segment["reduced_embedding"]
            cluster = segment["cluster"]
            annotation = segment["annotation"]

            csv_writer.writerow([index, data, embedding, cluster, annotation])


def load_plot(plot_file, start=0, end=None):
    with open(plot_file, "r") as file:
        plot_data = json.load(file)
    return plot_data[start:end]


def save_plot_json(plot_data, plot_file):
    with open(plot_file, "w") as file:
        json.dump(plot_data, file)


def get_plot_file(dataset_name: str, model_name: str, suffix="json"):
    directory = get_supervised_path("plot", dataset_name, model_name)
    # plot_directory = get_model_file_path("plot", dataset_name, model_name, )

    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"plot_{dataset_name}.{suffix}")
