from fastapi import APIRouter, Depends
from bertopic import BERTopic
from data.schemas import Dataset_names, Experimental_dataset_names
from models.service import load_model
from plot.service import get_plot

router = APIRouter()


@router.get("/plot/{dataset_name}")
def get_plot_endpoint(dataset_name: Experimental_dataset_names, model: BERTopic = Depends(load_model)):
    return {"data": get_plot(dataset_name)}
