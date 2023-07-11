from fastapi import APIRouter, Depends
from bertopic import BERTopic

from models.service import ModelService
from data.schemas import Dataset_names, Experimental_dataset_names
from models.schemas import Model_names
from embeddings.service import get_embeddings, get_reduced_embeddings, extract_embeddings, extract_embeddings_reduced

router = APIRouter()


@router.get("/")
def get_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 2):
    return {"embeddings": get_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}


@router.get("/reduced")
def get_reduced_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100):
    return {"reduced_embeddings": get_reduced_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}


@router.get("/extract")
def extract_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    extract_embeddings(dataset_name, model_name)
    return {"message": "Embeddings extracted successfully"}


@router.get("/extract/reduced")
def extract_embeddings_reduced_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    extract_embeddings_reduced(dataset_name, model_name)
    return {"message": "Reduced embeddings extracted successfully"}
