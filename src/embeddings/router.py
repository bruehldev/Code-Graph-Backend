from fastapi import APIRouter, Depends
from bertopic import BERTopic
from models.service import load_model
from data.schemas import Dataset_names, Experimental_dataset_names
from embeddings.service import get_embeddings

router = APIRouter()


@router.get("/load_model/{dataset_name}")
def load_model_endpoint(dataset_name: Experimental_dataset_names, model: BERTopic = Depends(load_model)):
    return {"message": f"{dataset_name} dataset loaded successfully"}


@router.get("/embeddings/{dataset_name}")
def get_embeddings_endpoint(dataset_name: Experimental_dataset_names, embeddings: list = Depends(get_embeddings)):
    return {"embeddings": embeddings}
