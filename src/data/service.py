import requests
import zipfile
import json
from sklearn.datasets import fetch_20newsgroups
import os
import json
import logging
import pandas as pd
from database.postgresql import create, init_table, get_data as get_segments_db, table_has_entries, DataTable
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


def get_segments(dataset_name: str, start: int = 0, end: int = None):
    data_path_key = get_path_key("data", dataset_name)

    segments_data = None

    # Return data from database if it exists
    if table_has_entries(data_path_key, DataTable):
        data = get_segments_db(data_path_key, start, end, DataTable)
        return [row.__dict__ for row in data]
    else:
        segments_data = extract_segments(dataset_name)

    return segments_data[start:end]


def extract_segments(dataset_name: str, page=1, page_size=10, export_to_file=False):
    entries = []

    if dataset_name == "few_nerd":
        data_path_key = get_path_key("data", dataset_name)
        data_file_path = get_data_file_path(type="data", dataset_name=dataset_name, filename="train.txt")

        init_table(data_path_key, DataTable)

        if not os.path.exists(data_file_path):
            # Download the data if it doesn't exist
            download_few_nerd_dataset(dataset_name)

        os.makedirs(get_supervised_path("segments", dataset_name), exist_ok=True)
        # TODO Use get_data function but be careful with different data formats!!! It seems like that the second line is different when using get_data instead of the following code.
        with open(data_file_path, "r", encoding="utf8") as f:
            sentence = ""
            segment = ""
            segment_list = []
            cur_annotation = None
            position = 0
            total_entries = 0 if page_size is None else page_size

            # Calculate the total number of entries to process
            if page_size is None:
                for line in f:
                    if not line.strip():
                        total_entries += 1
            logger.info(f"Extracting segments dataset: {dataset_name} with total entries: {total_entries}")

            f.seek(0)  # Reset file pointer

            with tqdm(total=total_entries, desc=f"Extracting {dataset_name}") as pbar:
                for line in f:
                    line = line.strip()
                    if line:
                        word, annotation = line.split()
                        sentence += " " + word
                        if annotation != "O":
                            segment += " " + word
                            if annotation != cur_annotation:
                                cur_annotation = annotation
                        else:
                            if segment:
                                segment = segment.lstrip()
                                position = sentence.find(segment, position + 1)
                                segment_list.append((segment, cur_annotation, position))
                                segment = ""
                                cur_annotation = None
                    else:
                        for i in segment_list:
                            sentence = sentence.lstrip()
                            entry = {
                                "sentence": sentence,
                                "segment": i[0],
                                "annotation": i[1],
                                "position": i[2],
                            }
                            entries.append(entry)
                            create(data_path_key, DataTable, sentence, i[0], i[1], i[2])
                            pbar.update(1)
                        segment_list = []
                        sentence = ""
                        position = 0

                    if page_size is not None and len(entries) >= page_size:
                        break  # Stop processing if page_size is reached

    if export_to_file:
        save_segments(entries, get_segments_file(dataset_name))
        logger.info(f"Extracted and saved segments for dataset: {dataset_name}")

    return entries


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
