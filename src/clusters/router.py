from fastapi import APIRouter
from data.schemas import Experimental_dataset_names
from clusters.service import get_clusters


router = APIRouter()


@router.get("/clusters/{dataset_name}")
def get_clusters_endpoint(dataset_name: Experimental_dataset_names):
    return {"clusters": get_clusters(dataset_name)}
