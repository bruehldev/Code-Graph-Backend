from fastapi import APIRouter
from data.schemas import Experimental_dataset_names
from clusters.service import get_clusters
from models.schemas import Model_names


router = APIRouter()


@router.get("/data/{dataset_name}/model/{model_name}/clusters")
def get_clusters_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    return {"clusters": get_clusters(dataset_name, model_name)}
