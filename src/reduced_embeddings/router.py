from fastapi import APIRouter, HTTPException, status, Depends
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
from project.service import ProjectService
from db.session import get_db
from sqlalchemy.orm import Session
import pickle
from sqlalchemy import not_, and_, exists
from db.models import Embedding, Model, Project, ReducedEmbedding

import numpy as np

router = APIRouter()


@router.get("/")
def get_reduced_embeddings_endpoint(project_id: int, all: bool = False, page: int = 1, page_size: int = 100, db: Session = Depends(get_db)):
    reduced_embeddings = []
    project = ProjectService(project_id, db)
    model_entry = project.get_model_entry("reduction_config")
    return_dict = {}

    if all:
        reduced_embeddings = db.query(ReducedEmbedding).filter(ReducedEmbedding.model_id == model_entry.model_id).all()
    else:
        reduced_embeddings = (
            db.query(ReducedEmbedding).filter(ReducedEmbedding.model_id == model_entry.model_id).offset(page * page_size).limit(page_size).all()
        )
        return_dict.update({"page": page, "page_size": page_size})

    return_dict.update({"length": len(reduced_embeddings), "data": reduced_embeddings})

    return return_dict


@router.get("/extract")
def extract_embeddings_reduced_endpoint(
    project_id: int, all: bool = True, page: int = 1, page_size: int = 100, return_data: bool = True, db: Session = Depends(get_db)
):
    reduced_embeddings = []
    project: ProjectService = ProjectService(project_id, db)

    model_entry, reduction_model = project.get_model("reduction_config")
    embedding_hash = project.get_embedding_hash()
    embedding_model_entry = db.query(Model).filter(Model.model_hash == embedding_hash).first()

    # Main query to find embeddings
    embeddings_todo = (
        db.query(Embedding)
        .join(Model, Model.model_id == Embedding.model_id)
        .join(Project, Project.project_id == Model.project_id)
        .filter(and_(Project.project_id == project_id, Model.model_id == embedding_model_entry.model_id))
        .filter(not_(exists().where(and_(ReducedEmbedding.embedding_id == Embedding.embedding_id, ReducedEmbedding.model_id == model_entry.model_id))))
        .all()
    )
    if not len(embeddings_todo) == 0:
        embeddings_arrays = np.array([])
        embeddings_arrays = np.stack([pickle.loads(embedding.embedding_value) for embedding in embeddings_todo])
        if not reduction_model.fitted:
            reduction_model.fit(embeddings_arrays)
        reduced_embeddings = reduction_model.transform([pickle.loads(embedding.embedding_value) for embedding in embeddings_todo])
        position_mappings = [
            {"embedding_id": embedding.embedding_id, "model_id": model_entry.model_id, "pos_x": float(position_value[0]), "pos_y": float(position_value[1])}
            for position_value, embedding in zip(reduced_embeddings, embeddings_todo)
        ]
        db.bulk_insert_mappings(ReducedEmbedding, position_mappings)
        db.commit()
        project.save_model("reduction_config", reduction_model)

    return {"data": len(reduced_embeddings)}


"""
if all:
    reduced_embeddings = extract_embeddings_reduced(dataset_name, model_name)
    if return_data:
        return {"data": reduced_embeddings, "length": len(reduced_embeddings)}
else:
    reduced_embeddings = extract_embeddings_reduced(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)
    if return_data:
        return {"data": reduced_embeddings, "length": len(reduced_embeddings), "page": page, "page_size": page_size}

return {"data": [], "length": len(reduced_embeddings), "page": page, "page_size": page_size}
"""


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
