import json
import uvicorn
from fastapi import FastAPI
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
from plot.router import router as plot_router

from configmanager.service import ConfigManager

app = FastAPI()
app.include_router(data_router, prefix="/data")
app.include_router(model_router, prefix="/data/{dataset_name}/model")
app.include_router(embeddings_router, prefix="/data/{dataset_name}/model{model_name}/embeddings")
app.include_router(clusters_router, prefix="/data/{dataset_name}/model/{model_name}/clusters")
app.include_router(plot_router, prefix="/data/{dataset_name}/model/{model_name}/plot")
app.include_router(config_router, prefix="/config")


# Environment variables
env = {}
with open("../env.json") as f:
    env = json.load(f)

# Config manager
config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()


@app.get("/")
def read_root():
    return {"status": "online"}


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    uvicorn.run(app, host=env["host"], port=env["port"])
