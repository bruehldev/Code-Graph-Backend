import json
import logging
import os
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

from data.utils import get_data_file_path, get_root_path, get_supervised_path

logger = logging.getLogger(__name__)


def download_few_nerd_dataset(dataset_name: str):
    url = "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/main/data/supervised.zip"
    data_path = get_supervised_path("data", dataset_name)

    output_file = data_path + ".zip"
    os.makedirs(data_path, exist_ok=True)
    response = requests.get(url)
    with open(output_file, "wb") as file:
        file.write(response.content)
    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall(get_root_path("data", dataset_name))
    os.remove(output_file)


def get_segments(dataset_name: str, start: int = 0, end: int = None):
    segments_file = get_segments_file(dataset_name)
    segments_data = None
    if os.path.exists(segments_file):
        segments_data = load_segments(segments_file)
        logger.info(f"Loaded segments from file for dataset: {dataset_name}")
    else:
        logger.error(f"No segments for: {dataset_name}")
        # segments_data = extract_segments(dataset_name)
    return segments_data[start:end]


def load_segments(segments_file):
    with open(segments_file, "r") as file:
        segment_data = json.load(file)
    return segment_data


def save_segments_file(segments_data, segments_file: str):
    with open(segments_file, "w") as file:
        json.dump(segments_data, file)


def extract_sentences_and_annotations(dataset_name: str):
    sentences = []
    annotations = []

    if dataset_name == "few_nerd":
        with open(get_data_file_path(type="data", dataset_name=dataset_name, filename="train.txt"), "r", encoding="utf-8") as f:
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


def save_sentences(sentences_data, sentences_file: str):
    with open(sentences_file, "w") as file:
        json.dump(sentences_data, file)


def save_annotations(annotations_data, annotations_file: str):
    with open(annotations_file, "w") as file:
        json.dump(annotations_data, file)


def get_sentences(dataset_name: str, start: int = 0, end: int = None):
    sentences_file = get_sentences_file(dataset_name)
    sentences_data = None

    if os.path.exists(sentences_file):
        with open(sentences_file, "r") as file:
            sentences_data = json.load(file)
        logger.info(f"Loaded sentences from file for dataset: {dataset_name}")
    else:
        sentences_data, _ = extract_sentences_and_annotations(dataset_name)

    return sentences_data[start:end]


def get_annotations(dataset_name: str, start: int = 0, end: int = None):
    annotations_file = get_annotations_file(dataset_name)
    annotations_data = None

    if os.path.exists(annotations_file):
        with open(annotations_file, "r") as file:
            annotations_data = json.load(file)
        logger.info(f"Loaded annotations from file for dataset: {dataset_name}")
    else:
        _, annotations_data = extract_sentences_and_annotations(dataset_name)

    return annotations_data


def get_sentences_file(dataset_name: str):
    os.makedirs(get_supervised_path("sentences", dataset_name), exist_ok=True)
    return get_data_file_path(type="sentences", dataset_name=dataset_name, filename="sentences.json")


def get_annotations_file(dataset_name: str):
    os.makedirs(get_supervised_path("annotations", dataset_name), exist_ok=True)
    return get_data_file_path(type="annotations", dataset_name=dataset_name, filename="annotations.json")


def get_segments_file(dataset_name: str):
    os.makedirs(get_supervised_path("segments", dataset_name), exist_ok=True)
    return get_data_file_path(type="segments", dataset_name=dataset_name, filename="segments.json")
