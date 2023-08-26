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
    delete_embeddings,
)

router = APIRouter()


@router.get("/")
def get_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100, reduce_length: int = 3):
    embeddings = get_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size, with_index=True)
    # reduce the length of the embeddings
    if reduce_length is not None:
        n = reduce_length
        embeddings = [[index, embedding[:n]] for index, embedding in embeddings]

    return {"embeddings": embeddings}


@router.get("/extract")
def extract_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100, id=None):
    extract_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size, id=id)
    return {"message": "Embeddings extracted successfully"}


@router.delete("/")
def delete_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    deleted = delete_embeddings(dataset_name, model_name)
    return {"deleted": deleted}


@router.post("/")
def create_embedding_endpoint(embedding: List[float], dataset_name: Experimental_dataset_names, model_name: Model_names):
    id = None
    create_embedding(id, embedding, dataset_name, model_name)
    return {"message": "Embedding created successfully"}


@router.get("/{index}")
def read_embedding_endpoint(index: int, dataset_name: Experimental_dataset_names, model_name: Model_names):
    index, embedding = read_embedding(index, dataset_name, model_name)
    if embedding is not None:
        return {"index": index, "embedding": embedding}
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
