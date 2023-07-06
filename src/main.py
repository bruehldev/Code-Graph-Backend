import json
import torch
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
from clusters.service import *


from data.router import router as data_router
from models.router import router as model_router
from configmanager.router import router as config_router
from embeddings.router import router as embeddings_router
from clusters.router import router as clusters_router
from graph.router import router as graph_router

app = FastAPI()
app.include_router(data_router)
app.include_router(model_router)
app.include_router(config_router)
app.include_router(embeddings_router)
app.include_router(clusters_router)
app.include_router(graph_router)

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


if __name__ == "__main__":
    uvicorn.run(app, host=env["host"], port=env["port"])
