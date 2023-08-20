import requests
import zipfile
import json
from sklearn.datasets import fetch_20newsgroups
import os
import json
import logging
import pandas as pd
from database.postgresql import create, init_table, get_data as get_data_db, table_has_entries, SegmentsTable
from tqdm import tqdm
from data.utils import get_path_key, get_data_file_path, get_root_path, get_supervised_path
from data.file_operations import download_few_nerd_dataset, save_segments, get_segments_file

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
        annotations = extract_annotations_keys(dataset_name)

    return annotations


def save_annotations_keys(annotations: dict, annotations_file: str):
    with open(annotations_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4)


def get_annotations_keys_file(dataset_name: str):
    os.makedirs(get_supervised_path("annotations", dataset_name), exist_ok=True)
    return get_data_file_path(type="annotations", dataset_name=dataset_name, filename="annotations.json")


def load_annotations_keys(dataset_name: str):
    with open(get_data_file_path(type="annotations", dataset_name=dataset_name, filename="annotations.json"), "r") as f:
        annotations = json.load(f)
        return annotations


def extract_annotations_keys(dataset_name: str):
    annotations = {}

    if dataset_name == "few_nerd":
        data_file_path = get_data_file_path(type="data", dataset_name=dataset_name, filename="train.txt")
        if not os.path.exists(data_file_path):
            # Download the data if it doesn't exist
            download_few_nerd_dataset(dataset_name)
        key_id = 1
        with open(data_file_path, "r", encoding="utf8") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) > 1:
                    annotation = fields[1]
                    categories = annotation.split("-")

                    last_category = categories[-1]
                    last_category_parts = last_category.split("/")
                    categories = categories[:-1] + last_category_parts
                    nested_obj = annotations
                    for category in categories:
                        subcategories = nested_obj.setdefault(category, {})
                        subcategories.setdefault("id", key_id)
                        subcategories.setdefault("name", category)
                        subcategories.setdefault("subcategories", {})
                        nested_obj = subcategories.setdefault("subcategories", {})
                        key_id += 1

    annotations_file = get_annotations_keys_file(dataset_name)
    save_annotations_keys(annotations, annotations_file)
    logger.info(f"Extracted and saved annotations for dataset: {dataset_name}")

    return annotations
