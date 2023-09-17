import pickle
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends

from database.postgresql import get as get_in_db, create as create_in_db, update as update_in_db, delete as delete_in_db, get_embedding_table
from embeddings.service import get_embeddings, extract_embeddings

from data.schemas import Experimental_dataset_names
from models.schemas import Model_names
from embeddings.schemas import EmbeddingTable, EmbeddingEntry, DataEmbeddingResponse, EmbeddingData
from db.schemas import DeleteResponse
from db.models import Segment, Sentence, Dataset, Project, Embedding, Model

from models_neu.model_definitions import MODELS
from configmanager.service import ConfigManager
from db.session import get_db
from sqlalchemy import not_, and_, exists
from project.service import ProjectService
from sqlalchemy.orm import Session
from data.utils import get_path_key
from utilities.string_operations import generate_hash

router = APIRouter()


@router.get("/")
def get_embeddings_endpoint(
    project_id: int,
    all: bool = False,
    page: int = 1,
    page_size: int = 100,
    reduce_length: int = 3,
    db: Session = Depends(get_db),
):
    return_dict = {"reduced_length": reduce_length}
    embeddings = []
    project = ProjectService(project_id, db)
    model_entry = project.get_model_entry("embedding_config")

    if all:
        embeddings = db.query(Embedding).filter(Embedding.model_id == model_entry.model_id).all()
    else:
        embeddings = db.query(Embedding).filter(Embedding.model_id == model_entry.model_id).offset(page * page_size).limit(page_size).all()
        return_dict.update({"page": page, "page_size": page_size})

    result = limit_embeddings_length(
        [{"id": embedding.embedding_id, "embedding": pickle.loads(embedding.embedding_value).tolist()} for embedding in embeddings], reduce_length
    )
    return_dict.update({"length": len(result), "data": result})

    return return_dict


@router.get("/extract")
def extract_embeddings_endpoint(
    project_id: int = None,
    db: Session = Depends(get_db),
):
    print("Extracting embeddings")
    print(f"Project {project_id}")
    embeddings = []
    project = ProjectService(project_id, db)
    model_entry, embedding_model = project.get_model("embedding_config")
    subquery = exists().where(and_(Embedding.segment_id == Segment.segment_id, Embedding.model_id == model_entry.model_id))

    segments_and_sentences = (
        db.query(Segment, Sentence)
        .join(Sentence, Sentence.sentence_id == Segment.sentence_id)
        .join(Dataset, Dataset.dataset_id == Sentence.dataset_id)
        .join(Project, Project.project_id == Dataset.project_id)
        .filter(Project.project_id == project_id)
        .filter(not_(subquery))
        .all()
    )

    if not len(segments_and_sentences) == 0:
        segments, sentences = zip(*segments_and_sentences)
        embeddings = embedding_model.transform(segments, sentences)

        ## saving

        embedding_mappings = [
            {"segment_id": segment.segment_id, "model_id": model_entry.model_id, "embedding_value": pickle.dumps(embedding_value)}
            for embedding_value, segment in zip(embeddings, segments)
        ]

        # Bulk insert embeddings

        db.bulk_insert_mappings(Embedding, embedding_mappings)
        db.commit()
        project.save_model("embedding_config", embedding_model)

    return {"data": len(embeddings)}


"""
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
"""


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
