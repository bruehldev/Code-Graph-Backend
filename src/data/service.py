import json
from sklearn.datasets import fetch_20newsgroups
import os
import json
import logging
from data.utils import get_data_file_path, get_root_path, get_supervised_path
from data.file_operations import download_few_nerd_dataset
from codes.service import extract_codes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_data(dataset_name: str, start: int = 0, end: int = None) -> list:
    data_file_path = get_data_file_path(type="data", dataset_name=dataset_name, filename="train.txt")

    if dataset_name == "fetch_20newsgroups":
        data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
    elif dataset_name == "few_nerd":
        os.makedirs(get_root_path("data", dataset_name), exist_ok=True)

        # Download the data if it doesn't exist
        if not os.path.exists(data_file_path):
            download_few_nerd_dataset(dataset_name)

        # Return data from file if it exists
        with open(data_file_path, "r", encoding="utf8") as f:
            data = [doc.strip() for doc in f.readlines() if doc.strip()]

    else:
        return None

    logger.info(f"Loaded data from file for dataset: {dataset_name} with length: {len(data[start:end])}")
    return data[start:end]


### Annotation functions ###
# TODO Use DB and move file operations to file_operations.py
def get_annotations_keys(dataset_name: str):
    annotations_file = get_annotations_keys_file(dataset_name)
    annotations = None

    if os.path.exists(annotations_file):
        annotations = load_annotations_keys(dataset_name)
        logger.info(f"Loaded annotations from file for dataset: {dataset_name}")
    else:
        annotations = extract_codes(dataset_name)

    return annotations


def save_annotations_keys(dataset_name: str, annotations: dict):
    annotations_file = get_annotations_keys_file(dataset_name)

    with open(annotations_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4)


def get_annotations_keys_file(dataset_name: str):
    os.makedirs(get_supervised_path("annotations", dataset_name), exist_ok=True)
    return get_data_file_path(type="annotations", dataset_name=dataset_name, filename="annotations.json")


def load_annotations_keys(dataset_name: str):
    with open(get_data_file_path(type="annotations", dataset_name=dataset_name, filename="annotations.json"), "r") as f:
        annotations = json.load(f)
        return annotations
