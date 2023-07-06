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
import hdbscan
from data.service import *
from data.router import router as data_router
from configmanager.service import ConfigManager

app = FastAPI()
app.include_router(data_router)
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


def save_embeddings(embeddings: np.ndarray, file_name: str):
    with open(file_name, "w") as f:
        json.dump(embeddings.tolist(), f)


def load_embeddings(file_name: str) -> np.ndarray:
    with open(file_name, "r") as f:
        embeddings_list = json.load(f)
        return np.array(embeddings_list)


def load_positions(positions_file):
    with open(positions_file, "r") as file:
        positions_data = json.load(file)
    return positions_data


def save_positions(positions_data, positions_file):
    with open(positions_file, "w") as file:
        json.dump(positions_data, file)


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


def get_embeddings_file(dataset_name: str):
    embeddings_directory = os.path.join(env["embeddings_path"], dataset_name)
    os.makedirs(embeddings_directory, exist_ok=True)
    return os.path.join(embeddings_directory, f"embeddings_{dataset_name}.json")


def get_positions_file(dataset_name: str):
    positions_directory = os.path.join(env["positions_path"], dataset_name)
    os.makedirs(positions_directory, exist_ok=True)
    return os.path.join(positions_directory, f"positions_{dataset_name}.json")


def get_clusters_file(dataset_name: str):
    clusters_directory = os.path.join(env["clusters_path"], dataset_name)
    os.makedirs(clusters_directory, exist_ok=True)
    return os.path.join(clusters_directory, f"clusters_{dataset_name}.json")


def extract_embeddings(model, data):
    logger.info("Extracting embeddings for documents")
    embeddings = model._extract_embeddings(data)
    umap_model = umap.UMAP(**config.embedding_config.dict())
    return umap_model.fit_transform(embeddings)


def save_clusters(clusters: np.ndarray, file_name: str):
    with open(file_name, "w") as f:
        json.dump(clusters, f)


def load_clusters(file_name: str) -> np.ndarray:
    with open(file_name, "r") as f:
        clusters_list = json.load(f)
        return np.array(clusters_list)


@app.get("/load_model/{dataset_name}")
def load_model_endpoint(dataset_name: str, model: BERTopic = Depends(load_model)):
    logger.info(f"Model loaded successfully for dataset: {dataset_name}")
    return {"message": f"{dataset_name} dataset loaded successfully"}


@app.get("/topicinfo/{dataset_name}")
def get_topic_info(dataset_name: str, model: BERTopic = Depends(load_model)):
    logger.info(f"Getting topic info for dataset: {dataset_name}")
    topic_info = model.get_topic_info()
    return {"topic_info": topic_info.to_dict()}


@app.get("/embeddings/{dataset_name}")
def get_embeddings(dataset_name: str, model: BERTopic = Depends(load_model), data: list = Depends(get_data)):
    global embeddings_2d_bert
    embeddings_file = get_embeddings_file(dataset_name)

    if os.path.exists(embeddings_file):
        embeddings_2d_bert = load_embeddings(embeddings_file)
        logger.info(f"Loaded embeddings from file for dataset: {dataset_name}")
    else:
        embeddings_2d_bert = extract_embeddings(model, data)
        save_embeddings(embeddings_2d_bert, embeddings_file)
        logger.info(f"Computed and saved embeddings for dataset: {dataset_name}")

    return embeddings_2d_bert.tolist()


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


@app.get("/clusters/{dataset_name}")
def get_clusters(dataset_name: str, model: BERTopic = Depends(load_model), embeddings: list = Depends(get_embeddings)):
    logger.info(f"Getting clusters for dataset: {dataset_name}")
    clusters_file = get_clusters_file(dataset_name)

    if os.path.exists(clusters_file):
        clusters = load_clusters(clusters_file)
        clusters = np.atleast_1d(clusters)
        logger.info(f"Loaded clusters from file for dataset: {dataset_name}")
        # TODO Fix clusters not being a list but a string
    else:
        clusterer = hdbscan.HDBSCAN(**config.cluster_config.dict())
        clusters = clusterer.fit_predict(embeddings)
        # convert the clusters to a JSON serializable format
        clusters = [int(c) for c in clusterer.labels_]
        # serialize the clusters to JSON
        json_clusters = json.dumps(clusters)
        save_clusters(json_clusters, clusters_file)
        logger.info(f"Computed and saved clusters for dataset: {dataset_name}")

    return {"clusters": list(clusters)}


if __name__ == "__main__":
    uvicorn.run(app, host=env["host"], port=env["port"])
