from fastapi import APIRouter, Depends
import json
from bertopic import BERTopic
import logging
from models.service import load_model, get_topic_info
from data.schemas import Dataset_names, Experimental_dataset_names

router = APIRouter()


@router.get("/load_model/{dataset_name}")
def load_model_endpoint(dataset_name: Experimental_dataset_names, model: BERTopic = Depends(load_model)):
    return {"message": f"{dataset_name} dataset loaded successfully"}


@router.get("/topicinfo/{dataset_name}")
def get_topic_info_endpoint(dataset_name: Experimental_dataset_names, model: BERTopic = Depends(load_model)):
    return {"topic_info": get_topic_info().to_dict()}
