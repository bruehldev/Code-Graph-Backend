from fastapi import APIRouter, Depends
from bertopic import BERTopic
from models.service import load_model
from data.schemas import Dataset_names
from clusters.service import get_clusters


router = APIRouter()


@router.get("/clusters/{dataset_name}")
def get_clusters_endpoint(dataset_name: Dataset_names, model: BERTopic = Depends(load_model)):
    return {"clusters": get_clusters(dataset_name, model)}
