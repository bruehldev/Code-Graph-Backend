import json
import os
import json
import logging
import csv

from reduced_embeddings.service import get_reduced_embeddings
from segments.service import get_segments
from clusters.service import get_clusters
from utilities.string_operations import get_project_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def extract_plot(project_id: str, plots):
    # Convert index, data, embedding, and topic to JSON structure
    json_plot_file = get_plot_file(project_id, suffix="json")
    csv_plot_file = get_plot_file(project_id, suffix="csv")

    save_plot_json(plots, json_plot_file)
    save_plot_csv(plots, csv_plot_file)
    return plots


def save_plot_csv(plot_data, plot_file):
    with open(plot_file, "w", newline="") as file:
        csv_writer = csv.writer(file)

        # Write header row
        header = ["id", "sentence", "segment", "cluster", "x", "y", "code"]
        csv_writer.writerow(header)

        # Write plot data rows
        for plot in plot_data:
            csv_writer.writerow(
                [
                    int(plot["id"]),
                    plot["sentence"],
                    plot["segment"],
                    plot["cluster"],
                    plot["reduced_embedding"]["x"],
                    plot["reduced_embedding"]["y"],
                    plot["code"],
                ]
            )


def load_plot(plot_file, start=0, end=None):
    with open(plot_file, "r") as file:
        plot_data = json.load(file)
    return plot_data[start:end]


def save_plot_json(plot_data, plot_file):
    with open(plot_file, "w") as file:
        json.dump(plot_data, file)


def get_plot_file(project_id, suffix="json"):
    directory = get_project_path(project_id=project_id, type="plots")
    # plot_directory = get_model_file_path("plot", dataset_name, model_name, )

    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"plot_{project_id}.{suffix}")
