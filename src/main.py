import json
import uvicorn
from fastapi import FastAPI
import logging

from data.service import *
from models.service import *
from configmanager.service import ConfigManager
from embeddings.service import *
from clusters.service import *
from database.postgresql import start_engine, stop_engine


from data.router import router as data_router
from database.router import router as database_router
from models.router import router as model_router
from configmanager.router import router as config_router
from embeddings.router import router as embeddings_router
from clusters.router import router as clusters_router
from plot.router import router as plot_router
from segments.router import router as segments_router
from reduced_embeddings.router import router as reduced_embeddings_router

from configmanager.service import ConfigManager

app = FastAPI(title="CodeGraph")
app.include_router(data_router, prefix="/data", tags=["data"])
app.include_router(database_router, prefix="/database", tags=["database"])
app.include_router(segments_router, prefix="/data/{dataset_name}/segments", tags=["segments"])
app.include_router(model_router, prefix="/data/{dataset_name}/model", tags=["model"])
app.include_router(embeddings_router, prefix="/data/{dataset_name}/model{model_name}/embeddings", tags=["embeddings"])
app.include_router(reduced_embeddings_router, prefix="/data/{dataset_name}/model/{model_name}/reduced_embeddings", tags=["reduced_embeddings"])
app.include_router(clusters_router, prefix="/data/{dataset_name}/model/{model_name}/clusters", tags=["clusters"])
app.include_router(plot_router, prefix="/data/{dataset_name}/model/{model_name}/plot", tags=["plot"])
app.include_router(config_router, prefix="/config", tags=["config"])


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


@app.on_event("startup")
async def startup():
    start_engine()


@app.on_event("shutdown")
async def shutdown():
    stop_engine()


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    uvicorn.run(app, host=env["host"], port=env["port"])
