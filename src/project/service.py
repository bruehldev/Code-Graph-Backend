import json
import os
import pickle
from sqlalchemy.orm import Session
from db.models import Project, Config, Model
from fastapi import HTTPException
from utilities.string_operations import generate_hash, get_file_path
from models_neu.model_definitions import MODELS
import logging

logger = logging.getLogger(__name__)


class ProjectService:
    def __init__(self, project_id: int, db: Session):
        self.project_id = project_id
        self.db = db

    def get_project(self):
        if self.project_id is None:
            raise HTTPException(status_code=400, detail="No project id provided")
        try:
            project = self.db.query(Project).filter(Project.project_id == self.project_id).first()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{str(e)}")
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    def get_project_config(self):
        project = self.get_project()
        config = self.db.query(Config).filter_by(config_id=project.config_id).first()
        return config

    def set_project_config(self, config_id):
        project = self.get_project()
        project.config_id = config_id
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project

    def get_embedding_model(self):
        config = self.get_project_config()
        model_name = config.config["embedding_config"]["model_name"]
        model_hash = generate_hash({"project_id": self.project_id, "model": config.config["embedding_config"]})
        model_entry = self.db.query(Model).filter(Model.model_hash == model_hash).first()
        # Check if model exists
        model_path = get_file_path(self.project_id, "models", f"{model_hash}.pkl")

        if not os.path.exists(model_path):
            embedding_model = MODELS[model_name](config.config["embedding_config"]["args"])
            logger.info(f"Created new model for project {self.project_id}/{model_hash}")
        else:
            with open(model_path, "rb") as f:
                embedding_model = pickle.load(f)
            logger.info(f"Loaded model from file {model_path}")

        if model_entry is None:
            model_entry = Model(project_id=self.project_id, model_hash=model_hash)
            self.db.add(model_entry)
            self.db.commit()
            self.db.refresh(model_entry)

        return model_entry, embedding_model

    def save_embedding_model(self, embedding_model):
        config = self.get_project_config()
        model_hash = generate_hash({"project_id": self.project_id, "model": config.config["embedding_config"]})
        model_path = get_file_path(self.project_id, "models", f"{model_hash}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(embedding_model, f)
        logger.info(f"Saved model to file {model_path}")
