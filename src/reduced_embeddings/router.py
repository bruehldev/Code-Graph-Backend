from fastapi import APIRouter, HTTPException, status
from typing import List
from data.schemas import Experimental_dataset_names
from models.schemas import Model_names
from reduced_embeddings.service import (
    get_reduced_embeddings,
    extract_embeddings_reduced,
)
from database.postgresql import get as get_in_db, create as create_in_db, update as update_in_db, delete as delete_in_db, get_reduced_embedding_table
from data.utils import get_path_key

from reduced_embeddings.schemas import ReducedEmbeddingTable, ReducedEmbeddingEntry, ReducedEmbeddingData, DataReducedEmbeddingResponse
from db.schemas import DeleteResponse

router = APIRouter()


@router.get("/")
def get_reduced_embeddings_endpoint(
    dataset_name: Experimental_dataset_names, model_name: Model_names, all: bool = False, page: int = 1, page_size: int = 100
) -> ReducedEmbeddingTable:
    if all:
        reduced_embeddings = get_reduced_embeddings(dataset_name, model_name)
        return {"data": reduced_embeddings, "length": len(reduced_embeddings)}
    else:
        reduced_embeddings = get_reduced_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)
        return {"data": reduced_embeddings, "length": len(reduced_embeddings), "page": page, "page_size": page_size}


@router.get("/extract")
def extract_embeddings_reduced_endpoint(
    dataset_name: Experimental_dataset_names, model_name: Model_names, all: bool = True, page: int = 1, page_size: int = 100, return_data: bool = True
) -> ReducedEmbeddingTable:
    reduced_embeddings = []
    if all:
        reduced_embeddings = extract_embeddings_reduced(dataset_name, model_name)
        if return_data:
            return {"data": reduced_embeddings, "length": len(reduced_embeddings)}
    else:
        reduced_embeddings = extract_embeddings_reduced(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)
        if return_data:
            return {"data": reduced_embeddings, "length": len(reduced_embeddings), "page": page, "page_size": page_size}

    return {"data": [], "length": len(reduced_embeddings), "page": page, "page_size": page_size}


@router.get("/{id}")
def get_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int,
) -> DataReducedEmbeddingResponse:
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)
    data = None
    try:
        data = get_in_db(reduced_embeddings_table, id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return {"data": data}


""" This is not allowed and does not work anymore since the reduced_embedding id must be a segment id!
@router.post("/")
def insert_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    data: ReducedEmbeddingData = {"reduced_embedding": [0.0, 0.0, 0.0, 0.0]},
) -> DataReducedEmbeddingResponse:
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)

    response = None
    try:
        response = create_in_db(reduced_embeddings_table, reduced_embedding=data.reduced_embedding)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if response is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create data")

    return {"data": response}
"""


@router.delete("/{id}", response_model=DeleteResponse)
def delete_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int = 0,
):
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)

    try:
        return {"id": id, "deleted": delete_in_db(reduced_embeddings_table, id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")


@router.put("/{id}")
def update_data_route(
    dataset_name: Experimental_dataset_names, model_name: Model_names, id: int = 0, data: ReducedEmbeddingData = {"reduced_embedding": [0.1, 0.1, 0.1, 0.1]}
) -> DataReducedEmbeddingResponse:
    reduced_embedding_table_name = get_path_key("reduced_embedding", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    reduced_embeddings_table = get_reduced_embedding_table(reduced_embedding_table_name, segment_table_name)

    response = None
    try:
        response = update_in_db(reduced_embeddings_table, id, data.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

    return {"data": response}
