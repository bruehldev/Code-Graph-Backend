import json
import os
import json
import logging
from fastapi import Depends
from typing import List
from data.service import get_data
import logging
import torch
from transformers import BertTokenizer, BertModel

from configmanager.service import ConfigManager
from bertopic import BERTopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = {}
with open("../env.json") as f:
    env = json.load(f)
config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()

models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
embeddings_2d_bert = None


def load_model(dataset_name: str, data: list = Depends(get_data)):
    global models
    if dataset_name in models:
        return models[dataset_name]

    model_path = os.path.join(env["model_path"], dataset_name)
    os.makedirs(model_path, exist_ok=True)

    if not os.path.exists(os.path.join(model_path, "BERTopic")):
        model = BERTopic(**config.model_config.dict())
        topics, probs = model.fit_transform(data)
        model.save(os.path.join(model_path, "BERTopic"))
        models[dataset_name] = model
        logger.info(f"Model trained and saved for dataset: {dataset_name}")
        return model
    else:
        model = BERTopic.load(os.path.join(model_path, "BERTopic"))
        models[dataset_name] = model
        logger.info(f"Loaded model from file for dataset: {dataset_name}")
        return model


def get_topic_info(dataset_name: str, model: BERTopic = Depends(load_model)):
    logger.info(f"Getting topic info for dataset: {dataset_name}")
    topic_info = model.get_topic_info()
    return topic_info
