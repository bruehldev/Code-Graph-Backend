from fastapi import APIRouter
from typing import List
from data.schemas import Experimental_dataset_names
from models.schemas import Model_names
from reduced_embeddings.service import (
    get_reduced_embeddings,
    extract_embeddings_reduced,
)

router = APIRouter()


@router.get("/")
def get_reduced_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100):
    return {"reduced_embeddings": get_reduced_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}


@router.get("/extract")
def extract_embeddings_reduced_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    extract_embeddings_reduced(dataset_name, model_name)
    return {"message": "Reduced embeddings extracted successfully"}
