import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import and_, exists, not_
from sqlalchemy.orm import Session

from db.models import Cluster, Model, Project, ReducedEmbedding
from db.session import get_db
from project.service import ProjectService

router = APIRouter()


class ClustersTableResponse(BaseModel):
    id: int
    cluster: int


@router.get("/extract")
def extract_clusters_endpoint(
    project_id: int, all: bool = False, page: int = 1, page_size: int = 100, return_data: bool = False, db: Session = Depends(get_db)
):
    clusters = []
    project: ProjectService = ProjectService(project_id, db)
    model_entry, cluster_model = project.get_model("cluster_config")
    reduction_hash = project.get_reduction_hash()
    reduction_model_entry = db.query(Model).filter(Model.model_hash == reduction_hash).first()

    reduced_embeddings_todo = (
        db.query(ReducedEmbedding)
        .join(Model, Model.model_id == ReducedEmbedding.model_id)
        .join(Project, Project.project_id == Model.project_id)
        .filter(and_(Project.project_id == project_id, Model.model_id == reduction_model_entry.model_id))
        .filter(not_(exists().where(and_(Cluster.reduced_embedding_id == ReducedEmbedding.reduced_embedding_id, Cluster.model_id == model_entry.model_id))))
        .all()
    )

    if not len(reduced_embeddings_todo) == 0:
        reduced_embeddings_arrays = np.stack([np.array([reduced_embedding.pos_x, reduced_embedding.pos_y]) for reduced_embedding in reduced_embeddings_todo])

        clusters = cluster_model.transform(reduced_embeddings_arrays)
        cluster_mappings = [
            {"reduced_embedding_id": reduced_embedding.reduced_embedding_id, "model_id": model_entry.model_id, "cluster": int(cluster)}
            for cluster, reduced_embedding in zip(clusters, reduced_embeddings_todo)
        ]

        db.bulk_insert_mappings(Cluster, cluster_mappings)
        db.commit()

    return_dict = {"extracted": len(reduced_embeddings_todo)}
    if return_data:
        if all:
            clusters = db.query(Cluster).filter(Cluster.model_id == model_entry.model_id).all()
        else:
            clusters = db.query(Cluster).filter(Cluster.model_id == model_entry.model_id).offset(page * page_size).limit(page_size).all()
            return_dict.update({"page": page, "page_size": page_size})
        return_dict.update({"length": len(clusters), "data": clusters})

    return return_dict


@router.get("/")
def get_clusters_endpoint(project_id: int, all: bool = False, page: int = 1, page_size: int = 100, db: Session = Depends(get_db)):
    clusters = []
    project = ProjectService(project_id, db)
    model_entry = project.get_model_entry("cluster_config")
    return_dict = {}

    if all:
        clusters = db.query(Cluster).filter(Cluster.model_id == model_entry.model_id).all()
    else:
        clusters = db.query(Cluster).filter(Cluster.model_id == model_entry.model_id).offset(page * page_size).limit(page_size).all()
        return_dict.update({"page": page, "page_size": page_size})

    return_dict.update({"length": len(clusters), "data": clusters})

    return return_dict
