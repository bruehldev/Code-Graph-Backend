import logging
from configmanager.schemas import ConfigModel, EmbeddingConfig, ClusterConfig, ModelConfig
from db.models import Config, Project
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, session: Session):
        self.configs = {}
        self.session = session

    def get_all_configs(self):
        configs = self.session.query(Config).all()
        return configs

    def get_config(self, id):
        config = self.session.query(Config).filter_by(config_id=id).first()
        return config

    def save_config(self, config):
        self.session.add(config)
        self.session.commit()

    def update_config(self, id, new_config):
        config = self.session.query(Config).filter_by(config_id=id).first()
        if config:
            config.name = new_config.name
            config.model_config = new_config.model_config
            config.embedding_config = new_config.embedding_config
            config.cluster_config = new_config.cluster_config
            self.session.commit()
        else:
            raise Exception(f"Config '{id}' not found")

    def delete_config(self, id):
        config = self.session.query(Config).filter_by(config_id=id).first()
        if config:
            self.session.delete(config)
            self.session.commit()
        else:
            raise Exception(f"Config '{id}' not found")

    def get_project_config(self, project_id):
        project = self.session.query(Project).filter_by(project_id=project_id).first()
        if project:
            config = self.session.query(Config).filter_by(config_id=project.config_id).first()
            return config
        else:
            raise Exception(f"Project '{project_id}' not found")

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
            cluster_config=ClusterConfig(min_cluster_size=2, metric="euclidean", cluster_selection_method="eom"),
            default_limit=None,
        )
