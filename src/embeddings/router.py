from fastapi import APIRouter, HTTPException, status
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
from embeddings.schemas import EmbeddingTable, EmbeddingEntry, DataEmbeddingResponse

router = APIRouter()


@router.get("/")
def get_embeddings_endpoint(
    dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100, reduce_length: int = 3
) -> EmbeddingTable:
    embeddings = get_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size, with_id=True)

    result = transform_embeddings_to_dict(embeddings, reduce_length=reduce_length)
    return {"length": len(result), "page": page, "page_size": page_size, "reduce_length": reduce_length, "data": result}


@router.get("/extract")
def extract_embeddings_endpoint(
    dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100, id=None, reduce_length: int = 3
) -> EmbeddingTable:
    embeddings = extract_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size, id=id, return_with_id=True)
    result = transform_embeddings_to_dict(embeddings, reduce_length=reduce_length)
    return {"length": len(result), "page": page, "page_size": page_size, "data": result, "reduce_length": reduce_length}


def transform_embeddings_to_dict(embeddings, reduce_length=None):
    # reduce the length of the embeddings
    if reduce_length is not None:
        n = reduce_length
        embeddings = [[index, embedding[:n]] for index, embedding in embeddings]
    result = [{"embedding": embedding, "id": data_id} for data_id, embedding in embeddings]
    return result


@router.delete("/")
def delete_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    deleted = delete_embeddings(dataset_name, model_name)
    if deleted is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return {"deleted": deleted}


@router.post("/")
def create_embedding_endpoint(embedding: List[float], dataset_name: Experimental_dataset_names, model_name: Model_names) -> DataEmbeddingResponse:
    id = None
    id, embedding = create_embedding(id, embedding, dataset_name, model_name)
    if id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return {"data": {"id": id, "embedding": embedding}}


@router.get("/{id}")
def read_embedding_endpoint(id: int, dataset_name: Experimental_dataset_names, model_name: Model_names) -> DataEmbeddingResponse:
    id, embedding = read_embedding(id, dataset_name, model_name)
    if id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

    return {"data": {"id": id, "embedding": embedding}}


@router.put("/{id}")
def update_embedding_endpoint(id: int, new_embedding: List[float], dataset_name: Experimental_dataset_names, model_name: Model_names) -> DataEmbeddingResponse:
    id, embedding = update_embedding(id, new_embedding, dataset_name, model_name)
    if id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

    return {"data": {"id": id, "embedding": embedding}}


@router.delete("/{id}")
def delete_embedding_endpoint(id: int, dataset_name: Experimental_dataset_names, model_name: Model_names):
    id = delete_embedding(id, dataset_name, model_name)
    if id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return {"data": {"id": id}}
