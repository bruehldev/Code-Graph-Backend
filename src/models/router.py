from fastapi import APIRouter, Depends
import json
from bertopic import BERTopic
import logging
from models.service import load_model
from data.schemas import Dataset_names

router = APIRouter()


@router.get("/load_model/{dataset_name}")
def load_model_endpoint(dataset_name: Dataset_names, model: BERTopic = Depends(load_model)):
    return {"message": f"{dataset_name} dataset loaded successfully"}
