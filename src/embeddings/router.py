from fastapi import APIRouter, Depends
from bertopic import BERTopic

from models.service import ModelService
from data.schemas import Dataset_names, Experimental_dataset_names
from models.schemas import Model_names
from embeddings.service import get_embeddings, get_reduced_embeddings

router = APIRouter()


@router.get("/")
def get_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    return {"embeddings": get_embeddings(dataset_name, model_name)}


# reduced embeddings
@router.get("/reduced")
def get_reduced_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    return {"reduced_embeddings": get_reduced_embeddings(dataset_name, model_name)}
