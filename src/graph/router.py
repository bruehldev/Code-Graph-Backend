from fastapi import APIRouter, Depends
from bertopic import BERTopic
from data.schemas import Dataset_names
from models.service import load_model
from graph.service import get_graph

router = APIRouter()


@router.get("/graph/{dataset_name}")
def get_graph_endpoint(dataset_name: Dataset_names, model: BERTopic = Depends(load_model)):
    return {"positions": get_graph(dataset_name)}
