from fastapi import APIRouter, Depends

from data.service import (
    get_data,
    download_few_nerd_dataset,
    extract_annotations_keys,
    extract_segments,
    get_segments,
    get_annotations_keys,
    get_sentences,
    get_annotations,
    extract_sentences_and_annotations,
)

from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names

router = APIRouter()


@router.get("/{dataset_name}", response_model=DataResponse)
def get_data_route(dataset_name: Experimental_dataset_names, page: int = 1, page_size: int = 100) -> list:
    data = get_data(dataset_name, start=(page - 1) * page_size, end=page * page_size)
    return DataResponse(data=data)


@router.get("/{dataset_name}/download")
def load_few_nerd_dataset_route(dataset_name: Dataset_names):
    download_few_nerd_dataset(dataset_name)
    return {"message": "Few NERD dataset loaded successfully"}


@router.get("/{dataset_name}/annotations-keys/extract")
def extract_annotations_route(dataset_name: Dataset_names):
    extract_annotations_keys(dataset_name)
    return {"message": "Annotations extracted successfully"}


@router.get("/{dataset_name}/segments/extract")
def extract_segments_route(dataset_name: Dataset_names):
    extract_segments(dataset_name)
    return {"message": "Segments extracted successfully"}


@router.get("/{dataset_name}/segments")
def get_segments_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100):
    segments = get_segments(dataset_name, start=(page - 1) * page_size, end=page * page_size)
    return {"segments": segments}


@router.get("/{dataset_name}/annotations-keys")
def get_annotations_keys_route(dataset_name: Dataset_names):
    annotations = get_annotations_keys(dataset_name)
    return {"annotations": annotations}


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
