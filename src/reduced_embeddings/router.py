from fastapi import APIRouter, HTTPException, status
from typing import List
from data.schemas import Experimental_dataset_names
from models.schemas import Model_names
from reduced_embeddings.service import (
    get_reduced_embeddings,
    extract_embeddings_reduced,
)
from pydantic import BaseModel
from database.schemas import Data
from database.postgresql import get, create, update, delete, get_reduced_embedding_table
from data.utils import get_path_key
from reduced_embeddings.schemas import Reduced_Embedding


router = APIRouter()


class ReducedEmbeddingTableResponse(BaseModel):
    id: int
    reduced_embeddings: List[float]
    segment_id: int


@router.get("/")
def get_reduced_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100):
    return {"reduced_embeddings": get_reduced_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}


@router.get("/extract")
def extract_embeddings_reduced_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    extract_embeddings_reduced(dataset_name, model_name)
    return {"message": "Reduced embeddings extracted successfully"}


@router.get("/{id}", response_model=ReducedEmbeddingTableResponse)
def get_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int,
):
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)
    data = None
    try:
        data = get(reduced_embedding_table_name, reduced_embeddings_table, id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return data


@router.post("/", response_model=Reduced_Embedding)
def insert_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    data: Reduced_Embedding = {"reduced_embeddings": [0.0, 0.0, 0.0, 0.0]},
):
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)

    create(reduced_embedding_table_name, reduced_embeddings_table, reduced_embeddings=data.reduced_embeddings)
    return data.__dict__


@router.delete("/{id}")
def delete_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int = 0,
):
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)

    try:
        return {"id": id, "deleted": delete(segment_table_name, reduced_embeddings_table, id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")


@router.put("/{id}", response_model=Reduced_Embedding)
def update_data_route(
    dataset_name: Experimental_dataset_names, model_name: Model_names, id: int = 0, data: Reduced_Embedding = {"reduced_embeddings": [0.1, 0.1, 0.1, 0.1]}
):
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)

    update(segment_table_name, reduced_embeddings_table, id, data.dict())
    return data
