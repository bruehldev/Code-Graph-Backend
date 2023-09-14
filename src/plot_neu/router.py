import logging
import pickle

import numpy as np
from fastapi import APIRouter, Body, Depends
from sqlalchemy import not_, and_, exists
from sqlalchemy.orm import Session, aliased

from configmanager.service import ConfigManager
from db import session, models
from models_neu.model_definitions import MODELS
from db.models import Embedding, Segment, Sentence, Dataset, Project, ReducedEmbedding
from db.session import get_db, SessionLocal
import time

from utilities.timer import Timer

router = APIRouter()
model_directory = "/home/efi/PycharmProjects/new_database_setup/api/models/model_files"
config_manager = ConfigManager
config = config_manager.get_default_model()  # getconfig(project_id) || get_default_config()


def get_model(model_data: ModelDefinition, project_id, username, db: Session = Depends(get_db), name_prefix=""):
    def get_model_db(model_name, project_id):
        print(model_name, project_id)
        return (
            db.query(models.CombinedModel)
            .join(models.Project, models.CombinedModel.ProjectID == models.Project.ProjectID)
            .filter(models.CombinedModel.ProjectID == project_id)
            .filter(models.CombinedModel.ModelName == model_name)
            .first()
        )

    name = model_data.name
    args = model_data.args
    model = MODELS[name](fitted=False, arguments=args)
    model_data.name = name_prefix + model.__str__()
    db_model = get_model_db(model_data.name, project_id)
    if not db_model:
        save_model(model_data, model, project_id, username, db)
        db_model = get_model_db(model_data.name, project_id)
    with open(f"{model_directory}/{db_model.ModelFile}", "rb") as f:
        model = pickle.load(f)
    return model, db_model


def update_model(model_data, model, project_id, user):
    filename = f"{user}_{project_id}_{model_data.name}.pkl".replace("-", "_").replace(" ", "_").replace("/", "_")
    with open(f"{model_directory}/{filename}", "wb") as f:
        pickle.dump(model, f)


def save_model(model_data, model, project_id, user, db: Session = Depends(get_db)):
    filename = f"{user}_{project_id}_{model_data.name}.pkl".replace("-", "_").replace(" ", "_").replace("/", "_")
    with open(f"{model_directory}/{filename}", "wb") as f:
        pickle.dump(model, f)
    db_model = models.CombinedModel(
        ProjectID=project_id,
        ModelName=model_data.name,
        ModelFile=filename,

    )
    db.add(db_model)
    db.commit()


@router.post("/create")
def create_view(
        project_id: int,
        db: Session = Depends(get_db)
):
    result = (
        db.query(models.Project)
        .filter(
            models.Project.ProjectID == project_id
        )
        .all()
    )
    if len(result) == 0:
        return {"error": "Project not found"}
    # initialize, so get the models necessary
    with Timer("Get embedding and reduction models"):
        embedding_model, db_e = get_model(embedding_model_data, project_id, current_user.username, db)
        reduction_model, db_r = get_model(reduction_model_data, project_id, current_user.username, db,
                                          name_prefix=db_e.ModelName + "_")

    # get any segments that don't have embeddings
    # get both the segments and their corresponding sentence
    with Timer("Get segments and sentences to embed"):
        subquery = exists().where(
            and_(
                Embedding.SegmentID == Segment.SegmentID,
                Embedding.ModelID == db_e.ModelID
            )
        )

        # Initialize the query for Segments and corresponding Sentences
        segments_and_sentences = db.query(Segment, Sentence). \
            join(Sentence, Sentence.SentenceID == Segment.SentenceID). \
            join(Dataset, Dataset.DatasetID == Sentence.DatasetID). \
            join(Project, Project.ProjectID == Dataset.ProjectID). \
            filter(Project.ProjectID == project_id). \
            filter(not_(subquery)). \
            all()
        segments, sentences = [], []
        logging.log(logging.INFO, f"Segments and Sentences: #{len(segments_and_sentences)}")
        if len(segments_and_sentences) > 0:
            segments, sentences = zip(*segments_and_sentences)
        del segments_and_sentences
    if len(segments) > 0:
        with Timer("Generate Embeddings"):
            if not hasattr(embedding_model, "fitted") or not embedding_model.fitted:
                embedding_model.fit(segments, sentences)
            embedding_values = embedding_model.transform(segments, sentences)
        with Timer("Insert Embeddings"):
            embedding_mappings = [
                {
                    "SegmentID": segment.SegmentID,
                    "ModelID": db_e.ModelID,
                    "EmbeddingValues": pickle.dumps(embedding_value)
                }
                for embedding_value, segment in zip(embedding_values, segments)
            ]

            # Bulk insert embeddings
            db.bulk_insert_mappings(models.Embedding, embedding_mappings)
            db.commit()
    del segments
    del sentences
    with Timer("Get embeddings to reduce"):
        subquery = (
            db.query(models.Position.EmbeddingID)
            .filter(models.Position.ModelID == db_r.ModelID)
            .subquery()
        )

        # Main query to find embeddings
        embeddings_todo = db.query(Embedding). \
            join(CombinedModel, CombinedModel.ModelID == Embedding.ModelID). \
            join(Project, Project.ProjectID == CombinedModel.ProjectID). \
            filter(
            and_(
                Project.ProjectID == project_id,
                CombinedModel.ModelID == db_e.ModelID
            )
        ). \
            filter(
            not_(
                exists().where(
                    and_(
                        Position.EmbeddingID == Embedding.EmbeddingID,
                        Position.ModelID == db_r.ModelID
                    )
                )
            )
        ).all()
        logging.log(logging.INFO, f"Embeddings: #{len(embeddings_todo)}")
        embeddings_arrays = np.array([])
        if len(embeddings_todo) > 0:
            embeddings_arrays = np.stack([pickle.loads(embedding.EmbeddingValues) for embedding in embeddings_todo])
    if len(embeddings_arrays) > 0:
        with Timer("Generate Positions"):
            if not hasattr(reduction_model, "fitted") or not reduction_model.fitted:
                reduction_model.fit(embeddings_arrays)
            position_values = reduction_model.transform(embeddings_arrays)
        with Timer("Insert Positions"):
            position_mappings = [
                {
                    "EmbeddingID": embedding.EmbeddingID,
                    "ModelID": db_r.ModelID,
                    "Posx": float(position_value[0]),
                    "Posy": float(position_value[1])
                }
                for position_value, embedding in zip(position_values, embeddings_todo)
            ]
            db.bulk_insert_mappings(models.Position, position_mappings)
            db.commit()

    with Timer("Create View"):
        new_view = models.View(
            ProjectID=project_id,
            EmbeddingModelID=db_e.ModelID,
            ReductionModelID=db_r.ModelID
        )
        db.add(new_view)
        db.commit()
    update_model(embedding_model_data, embedding_model, project_id, current_user.username)
    update_model(reduction_model_data, reduction_model, project_id, current_user.username)
    return_dictionary = {
        "ViewID": new_view.ViewID,
        "ProjectID": new_view.ProjectID,
        "EmbeddingModelID": new_view.EmbeddingModelID,
        "ReductionModelID": new_view.ReductionModelID
    }
    db.close()
    del embeddings_todo
    del embedding_model
    del reduction_model
    return return_dictionary