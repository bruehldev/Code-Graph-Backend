import os
import json
import torch
import umap
import uvicorn
import requests
import zipfile
import io
import numpy as np
from fastapi import Depends, FastAPI
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer, BertModel
import logging
import hdbscan
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import ast

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
models = {}
docs = None
embeddings_2d_bert = None


class ModelConfig(BaseModel):
    language: str
    top_n_words: int
    n_gram_range: tuple[int, int]
    min_topic_size: int
    nr_topics: Optional[int]
    low_memory: bool
    calculate_probabilities: bool
    seed_topic_list: Optional[Any]
    embedding_model: Optional[Any]
    umap_model: Optional[Any]
    hdbscan_model: Optional[Any]
    vectorizer_model: Optional[Any]
    ctfidf_model: Optional[Any]
    representation_model: Optional[Any]
    verbose: bool

class EmbeddingConfig(BaseModel):
    n_neighbors: int
    n_components: int
    metric: str
    random_state: int

class ClusterConfig(BaseModel):
    min_cluster_size: int
    metric: str
    cluster_selection_method: str

class ConfigModel(BaseModel):
    name: str = "default"
    model_config: ModelConfig
    embedding_config: EmbeddingConfig
    cluster_config: ClusterConfig


# Environment variables
env = {
    'model_path': '../models',
    'embeddings_path': '../embeddings',
    'clusters_path': '../clusters',
    'positions_path': '../positions',
    'segments_path': '../segments',
    'data_path': '../data',
    'host': '0.0.0.0',
    'port': 8000,
    'configs' : 'configs.json'
}

# Load configurations from file or use default if file does not exist
if os.path.exists(env["configs"]):
    with open(env["configs"], 'r') as f:
        config = json.load(f)

class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.configs = {}
        self.load_configs()

    def load_configs(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.configs = json.load(f)

    def save_configs(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.configs, f, indent=4)

config_manager = ConfigManager(env["configs"])

config = ConfigModel(
    name="default",
    model_config=ModelConfig(
        language="english",
        top_n_words=10,
        n_gram_range=(1, 1),
        min_topic_size=10,
        nr_topics=None,
        low_memory=False,
        calculate_probabilities=False,
        seed_topic_list=None,
        embedding_model=None,
        umap_model=None,
        hdbscan_model=None,
        vectorizer_model=None,
        ctfidf_model=None,
        representation_model=None,
        verbose=False
    ),
    embedding_config=EmbeddingConfig(
        n_neighbors=15,
        n_components=2,
        metric="cosine",
        random_state=42
    ),
    cluster_config=ClusterConfig(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom"
    )
)


@app.post("/config", response_model=ConfigModel)
def create_config(config: ConfigModel):
    config_manager.configs[config.name] = config.dict()
    config_manager.save_configs()
    return config


@app.get("/config/{name}")
def get_config(name: str):
    if name in config_manager.configs:
        return config_manager.configs[name]
    else:
        return {"message": f"Config '{name}' does not exist."}

@app.get("/configs")
def get_all_configs():
    return config_manager.configs

@app.put("/config/{name}")
def update_config(name: str, config: ConfigModel):
    if name in config_manager.configs:
        config_manager.configs[name] = config.dict()
        config_manager.save_configs()
        return {"message": f"Config '{name}' updated successfully."}
    else:
        return {"message": f"Config '{name}' does not exist."}

@app.delete("/config/{name}")
def delete_config(name: str):
    if name in config_manager.configs:
        del config_manager.configs[name]
        config_manager.save_configs()
        return {"message": f"Config '{name}' deleted successfully."}
    else:
        return {"message": f"Config '{name}' does not exist."}


# model_config source: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py
# embedding_config source: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
# cluster_config source: https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/hdbscan_.py

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

def load_few_nerd_dataset():
    url = "https://cloud.tsinghua.edu.cn/f/09265750ae6340429827/?dl=1"
    output_file = "supervised.zip"
    output_folder = "data/few_nerd"
    response = requests.get(url)
    with open(output_file, "wb") as file:
        file.write(response.content)
    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall(output_folder)

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

def save_segments(segments_data, segments_file: str):
    with open(segments_file, 'w') as file:
        json.dump(segments_data, file)

def load_model(dataset: str, docs: list = Depends(get_docs)):
    global models
    if dataset in models:
        return models[dataset]
    
    model_path = os.path.join(env['model_path'], dataset)
    os.makedirs(model_path, exist_ok=True)
    
    if not os.path.exists(os.path.join(model_path, 'BERTopic')):
        model = BERTopic(**config.model_config.dict())
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
    embeddings_directory = os.path.join(env['embeddings_path'], dataset)
    os.makedirs(embeddings_directory, exist_ok=True)
    return os.path.join(embeddings_directory, f"embeddings_{dataset}.json")

def get_positions_file(dataset: str):
    positions_directory = os.path.join(env['positions_path'], dataset)
    os.makedirs(positions_directory, exist_ok=True)
    return os.path.join(positions_directory, f"positions_{dataset}.json")

def get_segments_file(dataset: str):
    segments_directory = os.path.join(env['segments_path'], dataset)
    os.makedirs(segments_directory, exist_ok=True)
    return os.path.join(segments_directory, f"segments_{dataset}.json")

def get_clusters_file(dataset: str):
    clusters_directory = os.path.join(env['clusters_path'], dataset)
    os.makedirs(clusters_directory, exist_ok=True)
    return os.path.join(clusters_directory, f"clusters_{dataset}.json")

def extract_embeddings(model, docs):
    logger.info("Extracting embeddings for documents")
    embeddings = model._extract_embeddings(docs)
    umap_model = umap.UMAP(**config.embedding_config.dict())
    return umap_model.fit_transform(embeddings)

def extract_annotations(dataset: str):
    annotations = {}

    with open(f"./data/{dataset}/supervised/train.txt", "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) > 1:
                annotation = fields[1]
                categories = annotation.split("-")

                nested_obj = annotations
                for category in categories[:-1]:
                    nested_obj = nested_obj.setdefault(category, {})

                last_category = categories[-1]
                last_categories = last_category.split("/")
                for category in last_categories[:-1]:
                    nested_obj = nested_obj.setdefault(category, {})
                nested_obj.setdefault(last_categories[-1], {})

    with open(f"./data/{dataset}/annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4)

def extract_segments(dataset:str):
    load_few_nerd_dataset()
    with open(f"./data/{dataset}/supervised/train.txt", 'r', encoding='utf8') as f:
      sentence = ""
      segment = ""
      segment_list = []
      cur_annotation = None
      entries = []
      pos = 0
      for line in f:
        line = line.strip()
        if line:
          word, annotation = line.split()
          sentence += " " + word
          if annotation != 'O':
            segment += " " + word
            if annotation != cur_annotation:
              cur_annotation = annotation
              position = pos
          else:
            if segment:
              segment = segment.lstrip()
              segment_list.append((segment, cur_annotation, position))
              segment = ""
              cur_annotation = None
          pos = pos + 1
        else:
          for i in segment_list:
            sentence = sentence.lstrip()
            entry = {
                   "sentence": sentence,
                   "segment": i[0],
                   "annotation": i[1],
                   "position": i[2]
                    }
            entries.append(entry)
          pos = 0
          segment_list = []
          sentence = ""
    segments_file = get_segments_file(dataset)
    save_segments(entries, segments_file)


def save_clusters(clusters: np.ndarray, filename: str):
    with open(filename, 'w') as f:
        json.dump(clusters, f)

def load_clusters(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        clusters_list = json.load(f)
        return np.array(clusters_list)

def load_annotations(dataset: str):
    with open(os.path.join(env['data_path'], dataset, 'annotations.json'), 'r') as f:
        annotations = json.load(f)
        return annotations


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
        try:
            positions_dict = load_positions(positions_file)
            logger.info(f"Loaded positions from file for dataset: {dataset}")
        except Exception as e:
            logger.error(f"Error loading positions from file for dataset: {dataset}")
            logger.error(str(e))
            raise
    else:
        # Convert index, data, embedding, and topic to JSON structure
        positions_dict = {}
        for index, (embedding, doc) in enumerate(zip(embeddings[:20], docs[:20])): 
            position = {
                "data": doc,
                "position": embedding,
                "topic_index": np.array(model.transform([str(embedding)]))[0].tolist()
            }
            positions_dict[str(index)] = position

        save_positions(positions_dict, positions_file)
        logger.info(f"Saved positions to file for dataset: {dataset}")

    positions = list(zip(docs, positions_dict.values()))

    logger.info(f"Retrieved positions for dataset: {dataset}")
    return {"positions": positions}

@app.get("/clusters/{dataset}")
def get_clusters(dataset: str, model: BERTopic = Depends(load_model), embeddings: list = Depends(get_embeddings)):
    logger.info(f"Getting clusters for dataset: {dataset}")
    clusters_file = get_clusters_file(dataset)

    if os.path.exists(clusters_file):
        clusters = load_clusters(clusters_file)
        clusters = np.atleast_1d(clusters)
        logger.info(f"Loaded clusters from file for dataset: {dataset}")
        # TODO Fix clusters not being a list but a string
    else:
        clusterer = hdbscan.HDBSCAN(**config.cluster_config.dict())
        clusters = clusterer.fit_predict(embeddings)
        # convert the clusters to a JSON serializable format
        clusters = [int(c) for c in clusterer.labels_]
        # serialize the clusters to JSON
        json_clusters = json.dumps(clusters)
        save_clusters(json_clusters, clusters_file)
        logger.info(f"Computed and saved clusters for dataset: {dataset}")

    return {"clusters": list(clusters)}

@app.get("/annotations/{dataset}")
def get_annotations(dataset : str):
    # get only for few nerd
    all_annotations = load_annotations(dataset)
    return all_annotations
    


if __name__ == "__main__":
    uvicorn.run(app, host=env['host'], port=env['port'])
