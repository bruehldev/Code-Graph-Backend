import json
import logging
import os

from utilities.string_operations import get_project_path

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
            mapper[code.parent_code_id]["subcategories"][code.code_id] = mapper[code.code_id]
        else:
            category_tree[code.code_id] = mapper[code.code_id]
    return category_tree
