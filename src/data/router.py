from fastapi import APIRouter, Depends
from typing import List, Literal
from pydantic import BaseModel
from enum import Enum
import json

env = {}
with open('../env.json') as f:
    env = json.load(f)


from data.service import (
    get_data,
    load_few_nerd_dataset,
    extract_annotations,
    extract_segments,
    get_segments,
    get_annotations
)

router = APIRouter()

class DataResponse(BaseModel):
    data: List[str]

class Dataset_names(str, Enum):
    few_nerd = "few_nerd"
    fetch_20newsgroups = "fetch_20newsgroups"

@router.get("/")
def read_root():
    return {"Hello": "BERTopic API"}

@router.get("/data/{dataset_name}", response_model=DataResponse)
def get_data_route(dataset_name: Dataset_names, offset: int = 0, page_size: int = None) -> list:
    if page_size is None:
        page_size = env.get('default_limit')
    data = get_data(dataset_name) # get the whole data list
    data = data[offset:offset+page_size] # slice the data list according to offset and page_size
    return DataResponse(data=data)


@router.get("/download/{dataset_name}")
def load_few_nerd_dataset_route(dataset_name: Literal["few_nerd"]):
    load_few_nerd_dataset(dataset_name)
    return {"message": "Few NERD dataset loaded successfully"}

@router.get("/extract-annotations/{dataset_name}")
def extract_annotations_route(dataset_name: Literal["few_nerd"]):
    extract_annotations(dataset_name)
    return {"message": "Annotations extracted successfully"}

@router.get("/extract-segments/{dataset_name}")
def extract_segments_route(dataset_name: Literal["few_nerd"]):
    extract_segments(dataset_name)
    return {"message": "Segments extracted successfully"}

@router.get("/segments/{dataset_name}")
def get_segments_route(dataset_name: Literal["few_nerd"], offset: int = 0, page_size: int = None):
    if page_size is None:
        page_size = env.get('default_limit')
    segments = get_segments(dataset_name) # get the whole segments list
    segments = segments[offset:offset+page_size] # slice the segments list according to offset and page_size
    return {"segments": segments}

@router.get("/annotations/{dataset_name}")
def get_annotations_route(dataset_name: Literal["few_nerd"]):
    return {"annotations": get_annotations(dataset_name)}