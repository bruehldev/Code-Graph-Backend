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

# Configuration
CONFIG = {
    'model_path': 'models',
    'embeddings_path': 'embeddings',
    'clusters_path': 'clusters',
    'positions_path': 'positions',
    'host': '0.0.0.0',
    'port': 8000
}

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
    if not os.path.exists(model_path):
        model = BERTopic()
        topics, probs = model.fit_transform(docs)
        model.save(model_path)
        models[dataset] = model
        logger.info(f"Model trained and saved for dataset: {dataset}")
        return model
    else:
        model = BERTopic.load(model_path)
        models[dataset] = model
        logger.info(f"Loaded model from file for dataset: {dataset}")
        return model

def get_embeddings_file(dataset: str):
    return os.path.join(CONFIG['embeddings_path'], f"embeddings_{dataset}.json")

def get_positions_file(dataset: str):
    return os.path.join(CONFIG['positions_path'], f"positions_{dataset}.json")

def get_clusters_file(dataset: str):
    return os.path.join(CONFIG['clusters_path'], f"clusters_{dataset}.json")

def extract_embeddings(model, docs):
    logger.info("Extracting embeddings for documents")
    embeddings = model._extract_embeddings(docs)
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
    return umap_model.fit_transform(embeddings)

def to_serializable(value):
    if isinstance(value, np.int64):
        return int(value)
    return value

def save_clusters(clusters: list, filename: str):
    with open(filename, 'w') as f:
        json.dump(clusters, f, default=to_serializable)

def load_clusters(filename: str) -> list:
    with open(filename, 'r') as f:
        clusters_list = json.load(f)
        clusters_array = np.array(clusters_list)
        return clusters_array.tolist()

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
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')
        clusters = clusterer.fit_predict(embeddings)
        save_clusters(clusters.tolist(), clusters_file)
        logger.info(f"Computed and saved clusters for dataset: {dataset}")

    return {"clusters": list(clusters)}

if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG['host'], port=CONFIG['port'])
