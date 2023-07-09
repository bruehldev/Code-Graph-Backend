from fastapi import APIRouter, Depends
from bertopic import BERTopic

from models.service import ModelService
from data.schemas import Dataset_names, Experimental_dataset_names
from models.schemas import Model_names
from embeddings.service import get_embeddings

router = APIRouter()


@router.get("/data/{dataset_name}/model/{model_name}/embeddings")
def get_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    return {"embeddings": get_embeddings(dataset_name, model_name)}
