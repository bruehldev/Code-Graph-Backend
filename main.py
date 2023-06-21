import os
import json
import torch
import umap
import uvicorn
import numpy as np
from fastapi import Depends, FastAPI
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer, BertModel
import logging
import hdbscan

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
models = {}
docs = None
embeddings_2d_bert = None

class ConfigModel:
    def __init__(self, name: str, model_config: dict, embedding_config: dict, cluster_config: dict):
        self.name = name
        self.model_config = model_config
        self.embedding_config = embedding_config
        self.cluster_config = cluster_config


# Configuration
CONFIG = {
    'model_path': 'models',
    'embeddings_path': 'embeddings',
    'clusters_path': 'clusters',
    'positions_path': 'positions',
    'host': '0.0.0.0',
    'port': 8000
}

# model_config source: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py
# embedding_config source: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
# cluster_config source: https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/hdbscan_.py

default_config = ConfigModel(
    name="default",
    model_config={
        "language": "english",
        "top_n_words": 10,
        "n_gram_range": (1, 1),
        "min_topic_size": 10,
        "nr_topics": None,
        "low_memory": False,
        "calculate_probabilities": False,
        "seed_topic_list": None,
        "embedding_model": None,
        "umap_model": None,
        "hdbscan_model": None,
        "vectorizer_model": None,
        "ctfidf_model": None,
        "representation_model": None,
        "verbose": False
    },
    embedding_config={"n_neighbors": 15, "n_components": 2, "metric": "cosine", "random_state": 42},
    cluster_config={"min_cluster_size": 15, "metric": "euclidean", "cluster_selection_method": "eom"}
)

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_docs(dataset: str) -> list:
    if dataset == "fetch_20newsgroups":
        return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    elif dataset == "few_nerd":
        with open('data/few_nerd/train.txt', 'r', encoding='utf8') as f:
            return [doc.strip() for doc in f.readlines() if doc.strip()]
    return None

def save_embeddings(embeddings: np.ndarray, filename: str):
    with open(filename, 'w') as f:
        json.dump(embeddings.tolist(), f)

def load_embeddings(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        embeddings_list = json.load(f)
        return np.array(embeddings_list)

def load_positions(positions_file):
    with open(positions_file, 'r') as file:
        positions_data = json.load(file)
    return positions_data

def save_positions(positions_data, positions_file):
    with open(positions_file, 'w') as file:
        json.dump(positions_data, file)

def load_model(dataset: str, docs: list = Depends(get_docs)):
    global models
    if dataset in models:
        return models[dataset]
    
    model_path = os.path.join(CONFIG['model_path'], dataset)
    os.makedirs(model_path, exist_ok=True)
    
    if not os.path.exists(os.path.join(model_path, 'BERTopic')):
        model = BERTopic(**default_config.model_config)
        topics, probs = model.fit_transform(docs)
        model.save(os.path.join(model_path, 'BERTopic'))
        models[dataset] = model
        logger.info(f"Model trained and saved for dataset: {dataset}")
        return model
    else:
        model = BERTopic.load(os.path.join(model_path, 'BERTopic'))
        models[dataset] = model
        logger.info(f"Loaded model from file for dataset: {dataset}")
        return model

def get_embeddings_file(dataset: str):
    embeddings_directory = os.path.join(CONFIG['embeddings_path'], dataset)
    os.makedirs(embeddings_directory, exist_ok=True)
    return os.path.join(embeddings_directory, f"embeddings_{dataset}.json")

def get_positions_file(dataset: str):
    positions_directory = os.path.join(CONFIG['positions_path'], dataset)
    os.makedirs(positions_directory, exist_ok=True)
    return os.path.join(positions_directory, f"positions_{dataset}.json")

def get_clusters_file(dataset: str):
    clusters_directory = os.path.join(CONFIG['clusters_path'], dataset)
    os.makedirs(clusters_directory, exist_ok=True)
    return os.path.join(clusters_directory, f"clusters_{dataset}.json")

def extract_embeddings(model, docs):
    logger.info("Extracting embeddings for documents")
    embeddings = model._extract_embeddings(docs)
    umap_model = umap.UMAP(**default_config.embedding_config)
    return umap_model.fit_transform(embeddings)

def save_clusters(clusters: np.ndarray, filename: str):
    with open(filename, 'w') as f:
        json.dump(clusters, f)

def load_clusters(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        clusters_list = json.load(f)
        return np.array(clusters_list)

@app.get("/")
def read_root():
    return {"Hello": "BERTopic API"}

@app.get("/load_model/{dataset}")
def load_model_endpoint(dataset: str, model: BERTopic = Depends(load_model)):
    logger.info(f"Model loaded successfully for dataset: {dataset}")
    return {"message": f"{dataset} dataset loaded successfully"}

@app.get("/topicinfo/{dataset}")
def get_topic_info(dataset: str, model: BERTopic = Depends(load_model)):
    logger.info(f"Getting topic info for dataset: {dataset}")
    topic_info = model.get_topic_info()
    return {"topic_info": topic_info.to_dict()}

@app.get("/embeddings/{dataset}")
def get_embeddings(dataset: str, model: BERTopic = Depends(load_model), docs: list = Depends(get_docs)):
    global embeddings_2d_bert
    embeddings_file = get_embeddings_file(dataset)

    if os.path.exists(embeddings_file):
        embeddings_2d_bert = load_embeddings(embeddings_file)
        logger.info(f"Loaded embeddings from file for dataset: {dataset}")
    else:
        embeddings_2d_bert = extract_embeddings(model, docs)
        save_embeddings(embeddings_2d_bert, embeddings_file)
        logger.info(f"Computed and saved embeddings for dataset: {dataset}")

    return embeddings_2d_bert.tolist()

@app.get("/positions/{dataset}")
def get_positions(dataset: str, model: BERTopic = Depends(load_model), embeddings: list = Depends(get_embeddings), docs: list = Depends(get_docs)):
    positions_file = get_positions_file(dataset)

    if os.path.exists(positions_file):
        positions_list = load_positions(positions_file)
        logger.info(f"Loaded positions from file for dataset: {dataset}")
    else:
        positions_list = embeddings
        save_positions(positions_list, positions_file)
        logger.info(f"Saved positions to file for dataset: {dataset}")

    positions = list(zip(docs, positions_list))

    logger.info(f"Retrieved positions for dataset: {dataset}")
    return {"positions": positions}

@app.get("/clusters/{dataset}")
def get_clusters(dataset: str, model: BERTopic = Depends(load_model), embeddings: list = Depends(get_embeddings)):
    logger.info(f"Getting clusters for dataset: {dataset}")
    clusters_file = get_clusters_file(dataset)

    if os.path.exists(clusters_file):
        clusters = load_clusters(clusters_file)
        logger.info(f"Loaded clusters from file for dataset: {dataset}")
    else:
        clusterer = hdbscan.HDBSCAN(**default_config.cluster_config)
        clusters = clusterer.fit_predict(embeddings)
        # convert the clusters to a JSON serializable format
        clusters = [int(c) for c in clusterer.labels_]

        # serialize the clusters to JSON
        json_clusters = json.dumps(clusters)
        print(type(clusters))
        save_clusters(json_clusters, clusters_file)
        logger.info(f"Computed and saved clusters for dataset: {dataset}")

    return {"clusters": list(clusters)}

if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG['host'], port=CONFIG['port'])
