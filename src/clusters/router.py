import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import and_, exists, not_
from sqlalchemy.orm import Session

from clusters.schemas import (ClusterData, ClusterEntry, ClusterTable,
                              DataClusterResponse)
from clusters.service import extract_clusters, get_clusters
from data.schemas import Experimental_dataset_names
from data.utils import get_path_key
from database.postgresql import create as create_in_db
from database.postgresql import delete as delete_in_db
from database.postgresql import get as get_in_db
from database.postgresql import get_cluster_table
from database.postgresql import update as update_in_db
from db.models import Cluster, Model, Project, ReducedEmbedding
from db.schema import DeleteResponse
from db.session import get_db
from models.schemas import Model_names
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


@router.get("/{id}")
def get_cluster_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, id: int) -> DataClusterResponse:
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    data = None
    try:
        data = get_in_db(cluster_table, id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return {"data": data}


@router.put("/{id}")
def update_cluster_endpoint(
    dataset_name: Experimental_dataset_names, model_name: Model_names, id: int, data: ClusterData = {"cluster": -2}
) -> DataClusterResponse:
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)

    response = None
    try:
        response = update_in_db(cluster_table, id, data.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

    return {"data": response}


@router.delete("/{id}")
def delete_cluster_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, id: int) -> DeleteResponse:
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("segments", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    try:
        return {"id": id, "deleted": delete_in_db(cluster_table, id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
