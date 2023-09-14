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

from data.utils import get_path_key
from utilities.string_operations import generate_hash

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
    page: int = 1,
    page_size: int = 100,
    project_id: int = None,
    id=None,
    reduce_length: int = 3,
    return_data: bool = False,
    db=Depends(get_db),
):
    embeddings = []
    config_manager = ConfigManager(db)
    config = config_manager.get_project_config(project_id)
    config_as_json = json.loads(config.config)

    model_name = config_as_json["embedding_config"]["model_name"]
    model_hash = generate_hash({"project_id": project_id, "model": config_as_json["embedding_config"]})
    model_entry = db.query(Model).filter(Model.model_hash == model_hash).first()
    if model_entry is None:
        model_entry = Model(project_id=project_id, model_hash=model_hash)
        db.add(model_entry)
        db.commit()
        db.refresh(model_entry)

    embedding_model = MODELS[model_name](config_as_json["embedding_config"]["args"])

    subquery = exists().where(and_(Embedding.segment_id == Segment.segment_id, Embedding.model_id == 3))

    segments_and_sentences = (
        db.query(Segment, Sentence)
        .join(Sentence, Sentence.sentence_id == Segment.sentence_id)
        .join(Dataset, Dataset.dataset_id == Sentence.dataset_id)
        .join(Project, Project.project_id == Dataset.project_id)
        .filter(Project.project_id == project_id)
        .filter(not_(subquery))
        .all()
    )
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

    print(embeddings[0])
    return {"config": len(embeddings)}


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
