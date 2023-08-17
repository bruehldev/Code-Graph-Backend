import requests
import zipfile
import json
from sklearn.datasets import fetch_20newsgroups
import os
import json
import logging
import pandas as pd
from database.postgresql import insert_data, init_data_table, get_data_range, table_has_entries
from tqdm import tqdm
from data.utils import get_path_key, get_file_path, get_root_path, get_supervised_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_data(dataset_name: str, start: int = 0, end: int = None) -> list:
    data_file_path = get_file_path(type="data", dataset_name=dataset_name, filename="train.txt")

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

    print(len(data))

    return data[start:end]


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
    return get_file_path(type="annotations", dataset_name=dataset_name, filename="annotations.json")


def load_annotations_keys(dataset_name: str):
    with open(get_file_path(type="annotations", dataset_name=dataset_name, filename="annotations.json"), "r") as f:
        annotations = json.load(f)
        return annotations


def extract_annotations_keys(dataset_name: str):
    annotations = {}

    if dataset_name == "few_nerd":
        data_file_path = get_file_path(type="data", dataset_name=dataset_name, filename="train.txt")
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


def get_segments_file(dataset_name: str):
    os.makedirs(get_supervised_path("segments", dataset_name), exist_ok=True)
    return get_file_path(type="segments", dataset_name=dataset_name, model_name=None, filename="segments.json")


def get_segments(dataset_name: str, start: int = 0, end: int = None):
    segments_file = get_segments_file(dataset_name)
    data_path_key = get_path_key("data", dataset_name)

    segments_data = None

    # Return data from database if it exists
    if table_has_entries(data_path_key):
        data = get_data_range(data_path_key, start, end)
        return [row.__dict__ for row in data]
    if os.path.exists(segments_file):
        segments_data = load_segments(segments_file)
        logger.info(f"Loaded segments from file for dataset: {dataset_name}")
    else:
        segments_data = extract_segments(dataset_name)

    return segments_data[start:end]


def load_segments(segments_file):
    with open(segments_file, "r") as file:
        segment_data = json.load(file)
    return segment_data


def save_segments(segments_data, segments_file: str):
    with open(segments_file, "w") as file:
        json.dump(segments_data, file)


def extract_segments(dataset_name: str, page=1, page_size=10):
    entries = []

    if dataset_name == "few_nerd":
        data_path_key = get_path_key("data", dataset_name)
        data_file_path = get_file_path(type="data", dataset_name=dataset_name, filename="train.txt")

        init_data_table(data_path_key)

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
            print(f"Total entries: {total_entries}")

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
                            insert_data(data_path_key, sentence, i[0], i[1], i[2])
                            pbar.update(1)
                        segment_list = []
                        sentence = ""
                        position = 0

                    if page_size is not None and len(entries) >= page_size:
                        break  # Stop processing if page_size is reached

    save_segments(entries, get_segments_file(dataset_name))
    logger.info(f"Extracted and saved segments for dataset: {dataset_name}")

    return entries


def extract_sentences_and_annotations(dataset_name: str):
    sentences = []
    annotations = []

    if dataset_name == "few_nerd":
        with open(get_file_path(type="data", dataset_name=dataset_name, filename="train.txt"), "r", encoding="utf-8") as f:
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
    os.makedirs(get_supervised_path("sentences", dataset_name), exist_ok=True)
    return get_file_path(type="sentences", dataset_name=dataset_name, filename="sentences.json")


def get_annotations_file(dataset_name: str):
    os.makedirs(get_supervised_path("annotations", dataset_name), exist_ok=True)
    return get_file_path(type="annotations", dataset_name=dataset_name, filename="annotations.json")


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

    return annotations_data[start:end]


def save_sentences(sentences_data, sentences_file: str):
    with open(sentences_file, "w") as file:
        json.dump(sentences_data, file)


def save_annotations(annotations_data, annotations_file: str):
    with open(annotations_file, "w") as file:
        json.dump(annotations_data, file)
