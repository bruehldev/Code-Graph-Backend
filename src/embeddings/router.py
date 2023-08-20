from fastapi import APIRouter
from typing import List
from data.schemas import Experimental_dataset_names
from models.schemas import Model_names
from embeddings.service import (
    get_embeddings,
    extract_embeddings,
    create_embedding,
    read_embedding,
    update_embedding,
    delete_embedding,
)

router = APIRouter()


@router.get("/")
def get_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 2):
    return {"embeddings": get_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}


@router.get("/extract")
def extract_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    extract_embeddings(dataset_name, model_name)
    return {"message": "Embeddings extracted successfully"}


@router.post("/")
def create_embedding_endpoint(embedding: List[float], dataset_name: Experimental_dataset_names, model_name: Model_names):
    create_embedding(embedding, dataset_name, model_name)
    return {"message": "Embedding created successfully"}


@router.get("/{index}")
def read_embedding_endpoint(index: int, dataset_name: Experimental_dataset_names, model_name: Model_names):
    embedding = read_embedding(index, dataset_name, model_name)
    if embedding is not None:
        return {"embedding": embedding}
    else:
        return {"message": "Embedding not found"}


@router.put("/{index}")
def update_embedding_endpoint(index: int, new_embedding: List[float], dataset_name: Experimental_dataset_names, model_name: Model_names):
    update_embedding(index, new_embedding, dataset_name, model_name)
    return {"message": "Embedding updated successfully"}


@router.delete("/{index}")
def delete_embedding_endpoint(index: int, dataset_name: Experimental_dataset_names, model_name: Model_names):
    delete_embedding(index, dataset_name, model_name)
    return {"message": "Embedding deleted successfully"}
