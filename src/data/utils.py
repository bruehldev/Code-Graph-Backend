import os
import json

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_root_path(type: str, dataset_name: str):
    return os.path.join(env["exported_folder"], type, dataset_name)


def get_supervised_path(type: str, dataset_name: str, model_name=None):
    if model_name is None:
        return os.path.join(get_root_path(type, dataset_name), "supervised")
    else:
        return os.path.join(get_root_path(type, dataset_name), "supervised", model_name[:12])


def get_path_key(type: str, dataset_name: str, model_name=None):
    if model_name is None:
        return os.path.join(type, dataset_name, "supervised").replace("/", "_")
    else:
        return os.path.join(type, dataset_name, "supervised", model_name[:12]).replace("/", "_")


def get_data_file_path(type: str, dataset_name: str, filename: str):
    return os.path.join(get_supervised_path(type, dataset_name), filename)


def get_model_file_path(type: str, dataset_name: str, model_name, filename: str):
    return os.path.join(get_supervised_path(type, dataset_name, model_name), filename)
