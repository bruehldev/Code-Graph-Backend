from fastapi import APIRouter, Depends
import json

env = {}
with open("../env.json") as f:
    env = json.load(f)

from data.service import (
    get_data,
    load_few_nerd_dataset,
    extract_annotations_keys,
    extract_segments,
    get_segments,
    get_annotations_keys
)

from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names

router = APIRouter()


@router.get("/data/{dataset_name}", response_model=DataResponse)
def get_data_route(dataset_name: Experimental_dataset_names, offset: int = 0, page_size: int = None) -> list:
    if page_size is None:
        page_size = env.get("default_limit")
    data = get_data(dataset_name, offset=offset, page_size=page_size)
    return DataResponse(data=data)


@router.get("/download/{dataset_name}")
def load_few_nerd_dataset_route(dataset_name: Dataset_names):
    load_few_nerd_dataset(dataset_name)
    return {"message": "Few NERD dataset loaded successfully"}


@router.get("/extract-annotations-keys/{dataset_name}")
def extract_annotations_route(dataset_name: Dataset_names):
    extract_annotations_keys(dataset_name)
    return {"message": "Annotations extracted successfully"}


@router.get("/extract-segments/{dataset_name}")
def extract_segments_route(dataset_name: Dataset_names):
    extract_segments(dataset_name)
    return {"message": "Segments extracted successfully"}


@router.get("/segments/{dataset_name}")
def get_segments_route(dataset_name: Dataset_names, offset: int = 0, page_size: int = None):
    if page_size is None:
        page_size = env.get("default_limit")
    segments = get_segments(dataset_name, offset=offset, page_size=page_size)
    return {"segments": segments}


@router.get("/annotations-keys/{dataset_name}")
def get_annotations_keys_route(dataset_name: Dataset_names):
    annotations = get_annotations_keys(dataset_name)
    return {"annotations": annotations}
