import pickle
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status

from database.postgresql import get as get_in_db, create as create_in_db, update as update_in_db, delete as delete_in_db, get_embedding_table
from embeddings.service import get_embeddings, extract_embeddings

from data.schemas import Experimental_dataset_names
from models.schemas import Model_names
from embeddings.schemas import EmbeddingTable, EmbeddingEntry, DataEmbeddingResponse, EmbeddingData
from database.schemas import DeleteResponse

from data.utils import get_path_key

router = APIRouter()


@router.get("/")
def get_embeddings_endpoint(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    all: bool = False,
    page: int = 1,
    page_size: int = 100,
    reduce_length: int = 3,
) -> EmbeddingTable:
    embeddings = []
    if all:
        embeddings = limit_embeddings_length(get_embeddings(dataset_name, model_name), reduce_length)
        return {"length": len(embeddings), "data": embeddings, "reduce_length": reduce_length}
    else:
        embeddings = limit_embeddings_length(get_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size), reduce_length)
        return {"length": len(embeddings), "page": page, "page_size": page_size, "reduce_length": reduce_length, "data": embeddings}


@router.get("/extract")
def extract_embeddings_endpoint(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    all: bool = False,
    page: int = 1,
    page_size: int = 100,
    id=None,
    reduce_length: int = 3,
    return_data: bool = False,
) -> EmbeddingTable:
    embeddings = []
    if all:
        embeddings = extract_embeddings(dataset_name, model_name)
        if return_data:
            embeddings = limit_embeddings_length(embeddings, reduce_length)
            return {"length": len(embeddings), "data": embeddings, "reduce_length": reduce_length}
    else:
        embeddings = extract_embeddings(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size, id=id)
        embeddings = limit_embeddings_length(embeddings, reduce_length)
        if return_data:
            if id is None:
                return {"length": len(embeddings), "page": page, "page_size": page_size, "data": embeddings, "reduce_length": reduce_length}
            else:
                return {"length": len(embeddings), "id": id, "data": embeddings, "reduce_length": reduce_length}
    return {"length": len(embeddings), "page": page, "page_size": page_size, "reduce_length": reduce_length, "data": []}


def limit_embeddings_length(embeddings, reduce_length):
    embeddings = [{"id": embedding["id"], "embedding": embedding["embedding"][:reduce_length]} for embedding in embeddings]

    return embeddings


@router.get("/{id}")
def get_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int,
) -> DataEmbeddingResponse:
    embedding_table_name = get_path_key("embeddings", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    embeddings_table = get_embedding_table(embedding_table_name, segment_table_name)
    data = None
    try:
        data = get_in_db(embeddings_table, id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    data["embedding"] = pickle.loads(data["embedding"])
    return {"data": data}


@router.delete("/{id}", response_model=DeleteResponse)
def delete_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int,
):
    embedding_table_name = get_path_key("embeddings", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    embeddings_table = get_embedding_table(embedding_table_name, segment_table_name)
    data = None
    try:
        return {"id": id, "deleted": delete_in_db(embeddings_table, id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")


@router.put("/{id}")
def update_data_route(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    id: int,
    data: EmbeddingData = {"embedding": [0.1, 0.1, 0.1, 0.1]},
) -> DataEmbeddingResponse:
    embedding_table_name = get_path_key("embeddings", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    embeddings_table = get_embedding_table(embedding_table_name, segment_table_name)

    response = None
    try:
        embedding_data = data.dict()
        embedding_data = {"embedding": pickle.dumps(embedding_data["embedding"])}
        response = update_in_db(embeddings_table, id, embedding_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    response = {"id": response["id"], "embedding": pickle.loads(response["embedding"])}
    return {"data": response}
