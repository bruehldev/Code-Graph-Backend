import os
import json
import torch
import umap
import uvicorn
import numpy as np
from fastapi import Depends, FastAPI
from bertopic import BERTopic
from transformers import BertTokenizer, BertModel
import logging

from data.service import *
from models.service import *
from configmanager.service import ConfigManager
from embeddings.service import *


from data.router import router as data_router
from models.router import router as model_router
from configmanager.router import router as config_router
from embeddings.router import router as embeddings_router
from clusters.router import router as clusters_router


app = FastAPI()
app.include_router(data_router)
app.include_router(model_router)
app.include_router(config_router)
app.include_router(embeddings_router)
app.include_router(clusters_router)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
models = {}
embeddings_2d_bert = None


# Environment variables
env = {}

with open("../env.json") as f:
    env = json.load(f)

from configmanager.service import ConfigManager

config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


@app.get("/")
def read_root():
    return {"Hello": "BERTopic API"}


# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_positions(positions_file):
    with open(positions_file, "r") as file:
        positions_data = json.load(file)
    return positions_data


def save_positions(positions_data, positions_file):
    with open(positions_file, "w") as file:
        json.dump(positions_data, file)


def get_positions_file(dataset_name: str):
    positions_directory = os.path.join(env["positions_path"], dataset_name)
    os.makedirs(positions_directory, exist_ok=True)
    return os.path.join(positions_directory, f"positions_{dataset_name}.json")


@app.get("/positions/{dataset_name}")
def get_positions(dataset_name: str, model: BERTopic = Depends(load_model), embeddings: list = Depends(get_embeddings), data: list = Depends(get_data)):
    positions_file = get_positions_file(dataset_name)

    if os.path.exists(positions_file):
        try:
            positions_dict = load_positions(positions_file)
            logger.info(f"Loaded positions from file for dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading positions from file for dataset: {dataset_name}")
            logger.error(str(e))
            raise
    else:
        # Convert index, data, embedding, and topic to JSON structure
        positions_dict = {}
        for index, (embedding, doc) in enumerate(zip(embeddings[:20], data[:20])):
            position = {"data": doc, "position": embedding, "topic_index": np.array(model.transform([str(embedding)]))[0].tolist()}
            positions_dict[str(index)] = position

        save_positions(positions_dict, positions_file)
        logger.info(f"Saved positions to file for dataset: {dataset_name}")

    positions = list(zip(data, positions_dict.values()))

    logger.info(f"Retrieved positions for dataset: {dataset_name}")
    return {"positions": positions}


if __name__ == "__main__":
    uvicorn.run(app, host=env["host"], port=env["port"])
