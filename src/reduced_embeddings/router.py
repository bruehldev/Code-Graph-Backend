from fastapi import APIRouter
from typing import List
from data.schemas import Experimental_dataset_names
from models.schemas import Model_names
from reduced_embeddings.service import (
    get_reduced_embeddings,
    extract_embeddings_reduced,
)
from pydantic import BaseModel
from database.schemas import Data
from database.postgresql import get, create, update, delete, ReducedEmbeddingsTable
from data.utils import get_path_key
from reduced_embeddings.schemas import Reduced_Embedding


router = APIRouter()


class ReducedEmbeddingTableResponse(BaseModel):
    id: int
    reduced_embeddings: List[float]


@router.get("/")
def get_reduced_embeddings_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100):
    return {"reduced_embeddings": get_reduced_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}



@router.get("/{id}", response_model=ReducedEmbeddingTableResponse)
def get_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int,
):
    table_name = get_path_key("reduced_embedding", dataset_name, model_name)

    data = get(table_name, ReducedEmbeddingsTable, id)
    return data.__dict__


@router.post("/", response_model=Reduced_Embedding)
def insert_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    data: Reduced_Embedding = {"reduced_embeddings": [0.0, 0.0, 0.0, 0.0]},
):
    table_name = get_path_key("reduced_embedding", dataset_name, model_name)

    create(table_name, ReducedEmbeddingsTable, reduced_embeddings=data.reduced_embeddings)
    return data.__dict__


@router.delete("/{id}")
def delete_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int = 0,
):
    table_name = get_path_key("reduced_embedding", dataset_name, model_name)

    return {"id": id, "deleted": delete(table_name, ReducedEmbeddingsTable, id)}


@router.put("/{id}", response_model=Reduced_Embedding)
def update_data_route(
    dataset_name: Experimental_dataset_names, model_name: Model_names, id: int = 0, data: Reduced_Embedding = {"reduced_embeddings": [0.1, 0.1, 0.1, 0.1]}
):
    table_name = get_path_key("reduced_embedding", dataset_name, model_name)

    update(table_name, ReducedEmbeddingsTable, id, data.dict())
    return data


@router.get("/extract")
def extract_embeddings_reduced_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    extract_embeddings_reduced(dataset_name, model_name)
    return {"message": "Reduced embeddings extracted successfully"}