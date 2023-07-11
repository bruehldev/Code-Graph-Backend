from fastapi import APIRouter, Depends
from bertopic import BERTopic

from models.service import ModelService
from data.schemas import Dataset_names, Experimental_dataset_names
from models.schemas import Model_names
from embeddings.service import get_embeddings, get_reduced_embeddings

router = APIRouter()


@router.get("/")
def get_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 2):
    return {"embeddings": get_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}


# reduced embeddings
@router.get("/reduced")
def get_reduced_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100):
    return {"reduced_embeddings": get_reduced_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}
