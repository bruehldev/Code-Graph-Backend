from fastapi import APIRouter, Depends
from bertopic import BERTopic
from plot.service import get_plot
from data.schemas import Experimental_dataset_names
from models.schemas import Model_names

router = APIRouter()


@router.get("/")
def get_plot_endpoint(dataset_name: Experimental_dataset_names, model_names: Model_names):
    return {"plot": get_plot(dataset_name, model_names)}
