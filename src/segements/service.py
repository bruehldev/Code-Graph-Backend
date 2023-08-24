import json
import os
import json
import logging
from database.postgresql import create, init_table, get_data as get_data_db, table_has_entries, get_segment_table
from tqdm import tqdm
from data.utils import get_path_key, get_data_file_path, get_root_path, get_supervised_path
from data.file_operations import download_few_nerd_dataset, save_segments_file, get_segments_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_segments(dataset_name: str, start: int = 0, end: int = None):
    data_path_key = get_path_key("data", dataset_name)
    segment_table = get_segment_table(data_path_key)
    segments_data = None

    # Return data from database if it exists
    if table_has_entries(data_path_key, segment_table):
        data = get_data_db(data_path_key, start, end, segment_table)
        return data
    else:
        segments_data = extract_segments(dataset_name, start, end)

    return segments_data[start:end]


def extract_segments(dataset_name: str, start: int = 0, end: int = None, export_to_file: bool = False):
    entries = []

    if dataset_name == "few_nerd":
        data_file_path = get_data_file_path(type="data", dataset_name=dataset_name, filename="train.txt")

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
            total_entries = 0 if end is None else end
            limit = end
            # Calculate the total number of entries to process
            if end is None:
                for line in f:
                    if not line.strip():
                        total_entries += 1

            if start > 0:
                # If start is offset, we need to add the offset to limit
                limit = limit + start

            logger.info(f"Extracting segments dataset: {dataset_name} with total entries: {total_entries}. start: {start} end: {end} limit: {limit}")

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
                            pbar.update(1)
                        segment_list = []
                        sentence = ""
                        position = 0

                    if limit is not None and len(entries) >= limit:
                        # Important Note! Could be more than page_size if the last sentence has more than one annotation
                        # Each different annotation is counted as a new entry
                        break
    save_segments(entries, dataset_name, start, end)

    if export_to_file:
        save_segments_file(entries, get_segments_file(dataset_name))
        logger.info(f"Extracted and saved segments for dataset: {dataset_name}")

    return entries


def save_segments(entries, dataset_name: str, start=0, end=None):
    logger.info(f"Save segments in db: {dataset_name}. Length: {len(entries)}, start: {start}, end: {end}")
    segment_table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(segment_table_name)

    init_table(segment_table_name, segment_table)

    end = len(entries) if end is None else min(end, len(entries))  # Make sure end is within bounds

    for entry in entries[start:end]:
        create(
            segment_table_name, segment_table, sentence=entry["sentence"], segment=entry["segment"], annotation=entry["annotation"], position=entry["position"]
        )
