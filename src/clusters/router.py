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
from clusters.schemas import Cluster

router = APIRouter()


class ClustersTableResponse(BaseModel):
    id: int
    cluster: int


@router.get("/")
def list_clusters_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100):
    return {"clusters": get_clusters(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}


@router.get("/{id}", response_model=ClustersTableResponse)
def get_cluster_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, id: int):
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    data_from_db = get_in_db(cluster_table, id)

    if data_from_db is not None:
        return ClustersTableResponse(**data_from_db.__dict__)
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")


@router.post("/", response_model=Cluster)
def insert_cluster_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, data: Cluster = {"cluster": -1}):
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    create_in_db(cluster_table, cluster=data.cluster)
    return data.__dict__


@router.put("/{id}", response_model=Cluster)
def update_cluster_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, id: int, data: Cluster = {"cluster": -2}):
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    update_in_db(cluster_table, id, data.dict())
    return data


@router.delete("/{id}")
def delete_cluster_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, id: int):
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    return {"id": id, "deleted": delete_in_db(cluster_table, id)}


@router.get("/extract")
def extract_clusters_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    extract_clusters(dataset_name, model_name)
    return {"message": "Clusters extracted successfully"}
