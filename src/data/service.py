import requests
import zipfile
import json
from sklearn.datasets import fetch_20newsgroups
import os
import json
import logging
import pandas as pd

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

        if not os.path.exists(os.path.join(data_path, "supervised", "train.txt")):
            load_few_nerd_dataset(dataset_name)

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
        data = get_data(dataset_name)
        key_id = 1
        for line in data:
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


def extract_segments(dataset_name: str):
    entries = []

    if dataset_name == "few_nerd":
        segments_folder = os.path.join(env["segments_path"], dataset_name, "supervised")
        data_folder = os.path.join(env["data_path"], dataset_name)

        os.makedirs(segments_folder, exist_ok=True)
        # TODO Use get_data function but be careful with different data formats!!! It seems like that the second line is different when using get_data instead of the following code.
        with open(os.path.join(data_folder, "supervised", "train.txt"), "r", encoding="utf8") as f:
            sentence = ""
            segment = ""
            segment_list = []
            cur_annotation = None
            position = 0
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
                    segment_list = []
                    sentence = ""
                    position = 0

    save_segments(entries, get_segments_file(dataset_name))
    logger.info(f"Extracted and saved segments for dataset: {dataset_name}")

    return entries


def extract_sentences_and_annotations(dataset_name: str):
    sentences = []
    annotations = []

    if dataset_name == "few_nerd":
        data_folder = os.path.join(env["data_path"], dataset_name)
        with open(os.path.join(data_folder, "supervised", "train.txt"), "r", encoding="utf-8") as f:
            data = pd.read_csv(f, delimiter="\t", header=None, names=["sentence", "annotation"], skip_blank_lines=False)

            current_sentence = []
            current_annotation = []

            for row in data.itertuples(index=False):
                sentence, annotation = row.sentence, row.annotation
                if pd.isna(sentence):
                    if current_sentence:
                        sentences.append(" ".join(current_sentence))
                        annotations.append(" ".join(current_annotation))
                        current_sentence = []
                        current_annotation = []
                else:
                    current_sentence.append(sentence)
                    current_annotation.append(annotation)

            if current_sentence:
                sentences.append(" ".join(current_sentence))
                annotations.append(" ".join(current_annotation))

    save_sentences(sentences, get_sentences_file(dataset_name))
    save_annotations(annotations, get_annotations_file(dataset_name))

    logger.info(f"Extracted and saved sentences and annotations for dataset: {dataset_name}")

    return sentences, annotations


def get_sentences_file(dataset_name: str):
    sentences_directory = os.path.join(env["sentences_path"], dataset_name, "supervised")
    os.makedirs(sentences_directory, exist_ok=True)
    return os.path.join(sentences_directory, "sentences.json")


def get_annotations_file(dataset_name: str):
    annotations_directory = os.path.join(env["annotations_path"], dataset_name, "supervised")
    os.makedirs(annotations_directory, exist_ok=True)
    return os.path.join(annotations_directory, "annotations.json")


def get_sentences(dataset_name: str, offset: int = 0, page_size: int = None):
    sentences_file = get_sentences_file(dataset_name)
    sentences_data = None

    if os.path.exists(sentences_file):
        with open(sentences_file, "r") as file:
            sentences_data = json.load(file)
        logger.info(f"Loaded sentences from file for dataset: {dataset_name}")
    else:
        sentences_data, _ = extract_sentences_and_annotations(dataset_name)

    if page_size is not None:
        sentences_data = sentences_data[offset : offset + page_size]

    return sentences_data


def get_annotations(dataset_name: str, offset: int = 0, page_size: int = None):
    annotations_file = get_annotations_file(dataset_name)
    annotations_data = None

    if os.path.exists(annotations_file):
        with open(annotations_file, "r") as file:
            annotations_data = json.load(file)
        logger.info(f"Loaded annotations from file for dataset: {dataset_name}")
    else:
        _, annotations_data = extract_sentences_and_annotations(dataset_name)

    if page_size is not None:
        annotations_data = annotations_data[offset : offset + page_size]

    return annotations_data


def save_sentences(sentences_data, sentences_file: str):
    with open(sentences_file, "w") as file:
        json.dump(sentences_data, file)


def save_annotations(annotations_data, annotations_file: str):
    with open(annotations_file, "w") as file:
        json.dump(annotations_data, file)
