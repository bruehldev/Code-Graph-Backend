import pickle
import logging
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy import and_
from sqlalchemy.orm import Session, aliased
from tqdm import tqdm

from db.models import Cluster, Code, Embedding, Project, ReducedEmbedding, Segment, Sentence
from db.session import get_db
from dynamic.service import train_clusters
from embeddings.router import extract_embeddings_endpoint
from project.service import ProjectService
from reduced_embeddings.router import extract_embeddings_reduced_endpoint
from utilities.timer import Timer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/cluster")
def train_for_clusters(
    project_id: int,
    ids: List[int] = None,
    epochs: int = 10,
    db: Session = Depends(get_db),
):
    with Timer("setup"):
        project = ProjectService(project_id, db=db)
        embedding_model = project.get_model_entry("embedding_config")
        extract_embeddings_endpoint(project_id, db=db)
        embeddings = db.query(Embedding).filter(Embedding.model_id == embedding_model.model_id).all()
        # TODO currently only trains dynamic umap
        dyn_red_entry, dyn_red_model = project.get_model("reduction_config")
        if not hasattr(dyn_red_model, "is_dynamic") or getattr(dyn_red_model, "is_dynamic") == False:
            raise Exception("Currently only dynamic reduction is supported.")
    # get all embeddings, get all corresponding segment ids and all labels
    # should stay the same
    with Timer("query data"):
        EmbeddingAlias = aliased(Embedding)
        SegmentAlias = aliased(Segment)
        CodeAlias = aliased(Code)
        query = (
            db.query(SegmentAlias, EmbeddingAlias, CodeAlias)
            .join(SegmentAlias, EmbeddingAlias.segment_id == SegmentAlias.segment_id)
            .join(CodeAlias, SegmentAlias.code_id == CodeAlias.code_id)
            .filter(EmbeddingAlias.model_id == embedding_model.model_id)
        ).all()

    with Timer("create dataframe"):
        training_dicts = [
            {"id": segment.segment_id, "label": code.code_id, "embedding": pickle.loads(embedding.embedding_value)} for segment, embedding, code in query
        ]

        data = pd.DataFrame(training_dicts)
    for epoch in range(epochs):
        logger.info(f"Training epoch {epoch}")
        new_model = train_clusters(data, dyn_red_model, ids)
        dyn_red_model = new_model
        # recalculate reduced_embeddings and clusters TODO

    data_to_replace = db.query(ReducedEmbedding).filter(ReducedEmbedding.model_id == dyn_red_entry.model_id).all()
    logger.info("Deleting old reduced embeddings")
    db.query(Cluster).filter(
        Cluster.reduced_embedding_id.in_(db.query(ReducedEmbedding.embedding_id).filter(ReducedEmbedding.model_id == dyn_red_entry.model_id))
    ).delete(synchronize_session=False)
    db.query(ReducedEmbedding).filter(ReducedEmbedding.model_id == dyn_red_entry.model_id).delete(synchronize_session=False)
    db.commit()
    logger.info("Adding new reduced embeddings")
    project.save_model("reduction_config", dyn_red_model)
    extract_embeddings_reduced_endpoint(project_id, db=db)
    return True


@router.post("/correction")
def train_for_correction(
    project_id: int,
    correction: List[Dict[str, List[float]]] = None,
    epochs: int = 10,
    db: Session = Depends(get_db),
):
    with Timer("setup"):
        project = ProjectService(project_id, db=db)
        embedding_model = project.get_model_entry("embedding_config")
        extract_embeddings_endpoint(project_id, db=db)
        embeddings = db.query(Embedding).filter(Embedding.model_id == embedding_model.model_id).all()
        # TODO currently only trains dynamic umap
        dyn_red_entry, dyn_red_model = project.get_model("reduction_config")
        if not hasattr(dyn_red_model, "is_dynamic") or getattr(dyn_red_model, "is_dynamic") == False:
            raise Exception("Currently only dynamic reduction is supported.")
    # get all embeddings, get all corresponding segment ids and all labels
    # should stay the same
    with Timer("query data"):
        EmbeddingAlias = aliased(Embedding)
        SegmentAlias = aliased(Segment)
        CodeAlias = aliased(Code)
        query = (
            db.query(SegmentAlias, EmbeddingAlias, CodeAlias)
            .join(SegmentAlias, EmbeddingAlias.segment_id == SegmentAlias.segment_id)
            .join(CodeAlias, SegmentAlias.code_id == CodeAlias.code_id)
            .filter(EmbeddingAlias.model_id == embedding_model.model_id)
        ).all()

    with Timer("create dataframe"):
        training_dicts = [
            {"id": segment.segment_id, "label": code.code_id, "embedding": pickle.loads(embedding.embedding_value)} for segment, embedding, code in query
        ]

        data = pd.DataFrame(training_dicts)
    for epoch in range(epochs):
        logger.info(f"Training epoch {epoch}")
        new_model = train_points(data, dyn_red_model, correction)
        dyn_red_model = new_model
        # recalculate reduced_embeddings and clusters TODO

    data_to_replace = db.query(ReducedEmbedding).filter(ReducedEmbedding.model_id == dyn_red_entry.model_id).all()
    logger.info("Deleting old reduced embeddings")
    db.query(Cluster).filter(
        Cluster.reduced_embedding_id.in_(db.query(ReducedEmbedding.embedding_id).filter(ReducedEmbedding.model_id == dyn_red_entry.model_id))
    ).delete(synchronize_session=False)
    db.query(ReducedEmbedding).filter(ReducedEmbedding.model_id == dyn_red_entry.model_id).delete(synchronize_session=False)
    db.commit()
    logger.info("Adding new reduced embeddings")
    project.save_model("reduction_config", dyn_red_model)
    extract_embeddings_reduced_endpoint(project_id, db=db)
    return True
