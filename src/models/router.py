from fastapi import APIRouter, Depends
from models.service import ModelService
from models.schemas import Model_names
from data.schemas import Dataset_names, Experimental_dataset_names

router = APIRouter()


@router.get("/data/{dataset_name}/model/{model_name}")
def load_model_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    model_service = ModelService(dataset_name, model_name)
    return {"message": f"{dataset_name} dataset with {model_name} loaded successfully"}


@router.get("/data/{dataset_name}/model/{model_name}/topicinfo")
def get_topic_info_endpoint(dataset_name: Experimental_dataset_names, model_name: Model_names):
    model_service = ModelService(dataset_name, model_name)
    return {"topic_info": model_service.get_topic_info().to_dict()}
