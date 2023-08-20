from fastapi import APIRouter, Depends
from typing import List
from pydantic import BaseModel
from data.service import (
    get_data,
    extract_annotations_keys,
    get_annotations_keys,
)
from data.file_operations import download_few_nerd_dataset
from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names


router = APIRouter()


@router.get("/{dataset_name}")
def get_data_route(dataset_name: Experimental_dataset_names, page: int = 1, page_size: int = 100) -> list:
    data = get_data(dataset_name, start=(page - 1) * page_size, end=page * page_size)
    return data


@router.get("/{dataset_name}/download")
def load_few_nerd_dataset_route(dataset_name: Dataset_names):
    download_few_nerd_dataset(dataset_name)
    return {"message": "Few NERD dataset loaded successfully"}


@router.get("/{dataset_name}/annotations-keys/extract")
def extract_annotations_route(dataset_name: Dataset_names):
    extract_annotations_keys(dataset_name)
    return {"message": "Annotations extracted successfully"}


@router.get("/{dataset_name}/annotations-keys")
def get_annotations_keys_route(dataset_name: Dataset_names):
    annotations = get_annotations_keys(dataset_name)
    return {"annotations": annotations}


"""
@router.get("/{dataset_name}/sentences")
def get_sentences_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100):
    sentences = get_sentences(dataset_name, start=(page - 1) * page_size, end=page * page_size)
    return {"sentences": sentences}


@router.get("/{dataset_name}/annotations")
def get_annotations_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100):
    annotations = get_annotations(dataset_name, start=(page - 1) * page_size, end=page * page_size)
    return {"annotations": annotations}


@router.get("/{dataset_name}/sentences-and-annotations/extract")
def extract_sentences_and_annotations_route(dataset_name: Dataset_names):
    extract_sentences_and_annotations(dataset_name)
    return {"message": "Sentences and annotations extracted successfully"}
    """
