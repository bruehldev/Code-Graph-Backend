import json
import os
import json
import logging
from database.postgresql import (
    get_session,
    update_or_create,
    init_table,
    get_data as get_all_db,
    table_has_entries,
    get_code_table,
    get_all_codes,
    get_all_leaf_codes,
)
from tqdm import tqdm
from data.utils import get_path_key, get_data_file_path, get_root_path, get_supervised_path
from data.service import download_few_nerd_dataset

env = {}
with open("../env.json") as f:
    env = json.load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_codes(dataset_name: str):
    data_path_key = get_path_key("code", dataset_name)
    code_table = get_code_table(data_path_key)
    code_data = None

    if table_has_entries(code_table):
        code_data = get_all_db(code_table)
    else:
        code_data = extract_codes(dataset_name)

    return code_data


def get_top_level_codes(dataset_name: str):
    data_path_key = get_path_key("code", dataset_name)
    code_table = get_code_table(data_path_key)
    code_data = None

    if table_has_entries(code_table):
        code_data = get_all_codes(code_table)
    else:
        code_data = extract_codes(dataset_name)
        code_data = get_all_codes(code_table)

    return code_data


def get_leaf_codes(dataset_name: str):
    data_path_key = get_path_key("code", dataset_name)
    code_table = get_code_table(data_path_key)
    code_data = None

    if table_has_entries(code_table):
        code_data = get_all_leaf_codes(code_table)
    else:
        code_data = extract_codes(dataset_name)
        code_data = get_all_leaf_codes(code_table)

    return code_data


def save_codes(entries, dataset_name: str):
    code_table_name = get_path_key("code", dataset_name)
    code_table = get_code_table(code_table_name)
    init_table(code_table_name, code_table)
    stack = [(None, entries)]

    while stack:
        parent_id, code_data = stack.pop()
        session = get_session()
        for _, code_info in code_data.items():
            update_or_create(session, code_table, data_id=int(code_info["id"]), code=code_info["name"], top_level_code_id=parent_id)

            subcategories = code_info["subcategories"]
            if subcategories:
                stack.append((code_info["id"], subcategories))


def build_category_tree(categories_data, parent_id=None):
    category_tree = {}
    for category in categories_data:
        if category["top_level_code_id"] == parent_id:
            category_id = category["id"]
            category_name = category["code"]
            subcategories = build_category_tree(categories_data, parent_id=category_id)
            if subcategories:
                category_tree[category_name] = {
                    "id": category_id,
                    "name": category_name,
                    "subcategories": subcategories,
                }
            else:
                category_tree[category_name] = {
                    "id": category_id,
                    "name": category_name,
                    "subcategories": {},
                }
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
    save_codes(annotations, dataset_name)

    return annotations
