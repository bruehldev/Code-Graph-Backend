import requests
import zipfile
import json
from sklearn.datasets import fetch_20newsgroups
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_data(dataset_name: str, offset: int = 0, page_size: int = None) -> list:
    if dataset_name == "fetch_20newsgroups":
        data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
    elif dataset_name == "few_nerd":
        data_path = os.path.join(env["data_path"], dataset_name)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(os.path.join(data_path, "supervised", "train.txt"), "r", encoding="utf8") as f:
            data = [doc.strip() for doc in f.readlines() if doc.strip()]
    else:
        data = None
    if page_size is not None:
        data = data[offset : offset + page_size]  # slice the data list according to offset and page_size
    return data


def load_few_nerd_dataset(dataset_name: str):
    url = "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/main/data/supervised.zip"
    output_folder = os.path.join(env["data_path"], dataset_name)
    output_file = os.path.join(output_folder, "supervised.zip")
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    response = requests.get(url)
    with open(output_file, "wb") as file:
        file.write(response.content)
    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall(output_folder)
    os.remove(output_file)


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
    annotations_directory = os.path.join(env["annotations_keys_path"], dataset_name, "supervised")
    os.makedirs(annotations_directory, exist_ok=True)
    return os.path.join(annotations_directory, "annotations.json")


def load_annotations_keys(dataset_name: str):
    with open(os.path.join(env["annotations_keys_path"], dataset_name, "supervised", "annotations.json"), "r") as f:
        annotations = json.load(f)
        return annotations


def extract_annotations_keys(dataset_name: str):
    annotations = {}

    if dataset_name == "few_nerd":
        data_folder = os.path.join(env["data_path"], dataset_name)
        with open(os.path.join(data_folder, "supervised", "train.txt"), "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) > 1:
                    annotation = fields[1]
                    categories = annotation.split("-")

                    nested_obj = annotations
                    for category in categories[:-1]:
                        nested_obj = nested_obj.setdefault(category, {})

                    last_category = categories[-1]
                    last_categories = last_category.split("/")
                    for category in last_categories[:-1]:
                        nested_obj = nested_obj.setdefault(category, {})
                    nested_obj.setdefault(last_categories[-1], {})

    annotations_file = get_annotations_keys_file(dataset_name)
    save_annotations_keys(annotations, annotations_file)
    logger.info(f"Extracted and saved annotations for dataset: {dataset_name}")

    return annotations


def extract_segments(dataset_name: str):
    entries = []

    if dataset_name == "few_nerd":
        segments_folder = os.path.join(env["segments_path"], dataset_name, "supervised")
        data_folder = os.path.join(env["data_path"], dataset_name)

        os.makedirs(segments_folder, exist_ok=True)
        with open(os.path.join(data_folder, "supervised", "train.txt"), "r", encoding="utf8") as f:
            sentence = ""
            segment = ""
            segment_list = []
            cur_annotation = None
            pos = 0
            for line in f:
                line = line.strip()
                if line:
                    word, annotation = line.split()
                    sentence += " " + word
                    if annotation != "O":
                        segment += " " + word
                        if annotation != cur_annotation:
                            cur_annotation = annotation
                            position = pos
                    else:
                        if segment:
                            segment = segment.lstrip()
                            segment_list.append((segment, cur_annotation, position))
                            segment = ""
                            cur_annotation = None
                    pos = pos + 1
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
                    pos = 0
                    segment_list = []
                    sentence = ""

    save_segments(entries, get_segments_file(dataset_name))
    logger.info(f"Extracted and saved segments for dataset: {dataset_name}")

    return entries


def get_segments_file(dataset_name: str):
    segments_directory = os.path.join(env["segments_path"], dataset_name, "supervised")
    os.makedirs(segments_directory, exist_ok=True)
    return os.path.join(segments_directory, "segments.json")


def get_segments(dataset_name: str, offset: int = 0, page_size: int = None):
    segments_file = get_segments_file(dataset_name)
    segments_data = None

    if os.path.exists(segments_file):
        segments_data = load_segments(segments_file)
        logger.info(f"Loaded segments from file for dataset: {dataset_name}")
    else:
        segments_data = extract_segments(dataset_name)

    if page_size is not None:
        segments_data = segments_data[offset : offset + page_size]

    return segments_data


def load_segments(segments_file):
    with open(segments_file, "r") as file:
        segment_data = json.load(file)
    return segment_data


def save_segments(segments_data, segments_file: str):
    with open(segments_file, "w") as file:
        json.dump(segments_data, file)
