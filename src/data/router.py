from fastapi import APIRouter, Depends
from typing import List
from pydantic import BaseModel
import json

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

@router.get("/")
def read_root():
    return {"Hello": "BERTopic API"}

@router.get("/data/{dataset_name}", response_model=DataResponse)
def get_data_route(dataset_name: str) -> list:
    return DataResponse(data=get_data(dataset_name))

@router.post("/load-few-nerd-dataset")
def load_few_nerd_dataset_route():
    load_few_nerd_dataset()
    return {"message": "Few NERD dataset loaded successfully"}

@router.get("/extract-annotations/{dataset_name}")
def extract_annotations_route(dataset_name: str):
    extract_annotations(dataset_name)
    return {"message": "Annotations extracted successfully"}

@router.get("/extract-segments/{dataset_name}")
def extract_segments_route(dataset_name: str):
    extract_segments(dataset_name)
    return {"message": "Segments extracted successfully"}

@router.get("/segments/{dataset_name}")
def get_segments_route(dataset_name: str):
    return {"segments": get_segments(dataset_name)}

@router.get("/annotations/{dataset_name}")
def get_annotations_route(dataset_name: str):
    return {"annotations": get_annotations(dataset_name)}