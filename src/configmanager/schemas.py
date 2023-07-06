from pydantic import BaseModel
from enum import Enum
from typing import List
from typing import Dict, Any, Optional


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


# model_config source: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py
# embedding_config source: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
# cluster_config source: https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/hdbscan_.py
