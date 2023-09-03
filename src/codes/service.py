import json
import os
import json
import logging
from database.postgresql import create as create_in_db, init_table, get_data as get_all_db, table_has_entries, get_code_table, get_all_codes, get_all_leaf_codes
from tqdm import tqdm
from data.file_operations import get_annotations
from data.utils import get_path_key, get_data_file_path, get_root_path, get_supervised_path
from data.service import (
    get_data,
    extract_annotations_keys,
    get_annotations_keys,
)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_codes(dataset_name: str, start: int = 0, end: int = None):
    data_path_key = get_path_key("code", dataset_name)
    code_table = get_code_table(data_path_key)
    code_data = None

    if table_has_entries(code_table):
        code_data = get_all_db(code_table, start, end, True)
    else:
        code_data = get_annotations(dataset_name, start, end, True)
        save_codes(code_data, dataset_name)

    return code_data

def get_top_level_codes(dataset_name: str, start: int = 0, end: int = None):
    data_path_key = get_path_key("code", dataset_name)
    code_table = get_code_table(data_path_key)
    code_data = None

    if table_has_entries(code_table):
        code_data = get_all_codes(code_table)
    else:
        code_data = get_annotations(dataset_name, start, end, True)
        save_codes(code_data, dataset_name)
        code_data = get_all_codes(code_table)

    return code_data

def get_leaf_codes(dataset_name: str, start: int = 0, end: int = None):
    data_path_key = get_path_key("code", dataset_name)
    code_table = get_code_table(data_path_key)
    code_data = None

    if table_has_entries(code_table):
        code_data = get_all_leaf_codes(code_table)
    else:
        code_data = get_annotations(dataset_name, start, end, True)
        save_codes(code_data, dataset_name)
        code_data = get_all_leaf_codes(code_table)

    return code_data

def extract_codes(dataset_name: str, start: int = 0, end: int = None):

    code_data = get_annotations(dataset_name, start, end)
    save_codes(code_data, dataset_name)



def save_codes(entries, dataset_name: str):
    code_table_name = get_path_key("code", dataset_name)
    code_table = get_code_table(code_table_name)
    init_table(code_table_name, code_table)
    stack = [(None, entries)]

    while stack:
        parent_id, code_data = stack.pop()

        for _, code_info in code_data.items():
            create_in_db(code_table, id=code_info['id'], code=code_info['name'], top_level_code_id=parent_id)

            subcategories = code_info['subcategories']
            if subcategories:
                stack.append((code_info['id'], subcategories))

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