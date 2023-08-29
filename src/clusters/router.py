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
from database.schemas import DeleteResponse

router = APIRouter()


class ClustersTableResponse(BaseModel):
    id: int
    cluster: int


@router.get("/extract")
def extract_clusters_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100) -> ClusterTable:
    clusters = extract_clusters(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)
    return {"data": clusters, "length": len(clusters), "page": page, "page_size": page_size}


@router.get("/")
def list_clusters_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100) -> ClusterTable:
    clusters = get_clusters(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)
    return {"data": clusters, "length": len(clusters), "page": page, "page_size": page_size}


@router.get("/{id}")
def get_cluster_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, id: int) -> DataClusterResponse:
    cluster_table_name = get_path_key("clusters", dataset_name, model_name)
    segment_table_name = get_path_key("data", dataset_name)
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
    segment_table_name = get_path_key("data", dataset_name)
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
    segment_table_name = get_path_key("data", dataset_name)
    cluster_table = get_cluster_table(cluster_table_name, segment_table_name)
    try:
        return {"id": id, "deleted": delete_in_db(cluster_table, id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
