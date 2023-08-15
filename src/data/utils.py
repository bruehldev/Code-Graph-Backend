import os
import json

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_root_path(type: str, dataset_name: str):
    return os.path.join(env["exported_folder"], type, dataset_name)


def get_supervised_path(type: str, dataset_name: str):
    return os.path.join(get_root_path(type, dataset_name), "supervised")


def get_path_key(type: str, dataset_name: str):
    return os.path.join(type, dataset_name, "supervised").replace("/", "_")


def get_file_path(type: str, dataset_name: str, filename: str):
    data_file_path = os.path.join(get_supervised_path(type, dataset_name), filename)
    return data_file_path
