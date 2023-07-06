import requests
import zipfile
import json
from sklearn.datasets import fetch_20newsgroups
import os
import json
import logging
from configmanager.schemas import ConfigModel, ConfigModel, EmbeddingConfig, ClusterConfig, ModelConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


env = {}
with open("../env.json") as f:
    env = json.load(f)


class ConfigManager:
    config = None

    def __init__(self, config_file):
        # Load configurations from file or use default if file does not exist
        if os.path.exists(env["configs"]):
            with open(env["configs"], "r") as f:
                config = json.load(f)
        self.config_file = config_file
        self.configs = {}
        self.load_configs()

    def load_configs(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                self.configs = json.load(f)

    def save_configs(self):
        with open(self.config_file, "w") as f:
            json.dump(self.configs, f, indent=4)

    @staticmethod
    def get_default_model():
        return ConfigModel(
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
                verbose=False,
            ),
            embedding_config=EmbeddingConfig(n_neighbors=15, n_components=2, metric="cosine", random_state=42),
            cluster_config=ClusterConfig(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom"),
        )
