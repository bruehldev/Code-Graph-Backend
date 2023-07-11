from fastapi import APIRouter
from data.schemas import Experimental_dataset_names
from clusters.service import get_clusters, extract_clusters
from models.schemas import Model_names


router = APIRouter()


@router.get("/")
def get_clusters_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names, page: int = 1, page_size: int = 100):
    return {"clusters": get_clusters(dataset_name, model_name, start=(page - 1) * page_size, end=page * page_size)}


@router.get("/extract")
def extract_clusters_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    extract_clusters(dataset_name, model_name)
    return {"message": "Clusters extracted successfully"}
