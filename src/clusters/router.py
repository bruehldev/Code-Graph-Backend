from fastapi import APIRouter, HTTPException, status

from pydantic import BaseModel

from data.schemas import Experimental_dataset_names
from clusters.service import get_clusters, extract_clusters
from models.schemas import Model_names
from database.postgresql import (
    get_cluster_table,
    get as get_in_db,
    create as create_in_db,
    update as update_in_db,
    delete as delete_in_db,
)
from data.utils import get_path_key

from clusters.schemas import DataClusterResponse, ClusterTable, ClusterEntry, ClusterData
from db.schemas import DeleteResponse
from project.service import ProjectService
from db.session import get_db
from sqlalchemy.orm import Session
from fastapi import Depends
from db.models import Cluster, Model, Project, ReducedEmbedding
from sqlalchemy import not_, and_, exists

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
    cluster_hash = project.get_embedding_hash()
    cluster_model_entry = db.query(Model).filter(Model.model_hash == cluster_hash).first()

    reduced_embeddings_todo = (
        db.query(ReducedEmbedding)
        .join(Model, Model.model_id == ReducedEmbedding.model_id)
        .join(Project, Project.project_id == Model.project_id)
        .filter(and_(Project.project_id == project_id, Model.model_id == cluster_model_entry.model_id))
        .filter(not_(exists().where(and_(Cluster.reduced_embedding_id == ReducedEmbedding.reduced_embedding_id, Cluster.model_id == model_entry.model_id))))
        .all()
    )
    if not len(reduced_embeddings_todo) == 0:
        print(
            reduced_embeddings_todo[0].reduced_embedding_id,
            reduced_embeddings_todo[0].model_id,
            reduced_embeddings_todo[0].pos_x,
            reduced_embeddings_todo[0].pos_y,
        )
        cluster_model.transform(reduced_embeddings_todo)

    """
    if all:
        clusters = extract_clusters(dataset_name, model_name)
        if return_data:
            return {"data": clusters, "length": len(clusters)}
        else:
            return {"data": [], "length": len(clusters)}
    else:
        clusters = extract_clusters(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)
        if return_data:
            return {"data": clusters, "length": len(clusters), "page": page, "page_size": page_size}
        else:
            return {"data": [], "length": len(clusters), "page": page, "page_size": page_size}"""


@router.get("/")
def get_clusters_endpoint(
    dataset_name: Experimental_dataset_names, model_name: Model_names, all: bool = False, page: int = 1, page_size: int = 100
) -> ClusterTable:
    clusters = []
    if all:
        clusters = get_clusters(dataset_name, model_name)
        return {"data": clusters, "length": len(clusters)}
    else:
        clusters = get_clusters(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)
        return {"data": clusters, "length": len(clusters), "page": page, "page_size": page_size}


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
