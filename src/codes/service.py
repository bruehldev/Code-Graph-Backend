import json
import os
import json
import logging
from tqdm import tqdm
from data.utils import get_data_file_path
from data.service import download_few_nerd_dataset

env = {}
with open("../env.json") as f:
    env = json.load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_category_tree(codes):
    category_tree = {}
    mapper = {}
    for code in codes:
        temp_code = {
            "id": code.code_id,
            "name": code.text,
            "subcategories": {},
        }
        mapper[code.code_id] = temp_code
    for code in codes:
        if code.parent_code_id is not None:
            mapper[code.parent_code_id]["subcategories"][code.code_id]= mapper[code.code_id]
        else:
            category_tree[code.code_id] = mapper[code.code_id]
    return category_tree


def extract_codes(dataset_name: str):
    def add_category(nested_obj, category):
        nonlocal key_id
        if category not in nested_obj:
            subcategories = nested_obj.setdefault(category, {})
            subcategories["id"] = key_id
            subcategories["name"] = category
            subcategories["subcategories"] = {}
            nested_obj = subcategories["subcategories"]
            key_id += 1
        else:
            nested_obj = nested_obj[category]["subcategories"]
        return nested_obj

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
                        nested_obj = add_category(nested_obj, category)

    logger.info(f"Extracted and saved annotations for dataset: {dataset_name}")

    return annotations
