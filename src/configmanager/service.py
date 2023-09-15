import logging
from configmanager.schemas import ConfigModel, ReductionConfig, ClusterConfig, EmbeddingConfig
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
            config.embedding_config = new_config.embedding_config
            config.reduction_config = new_config.reduction_config
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

    @staticmethod
    def get_default_model():
        return ConfigModel()
