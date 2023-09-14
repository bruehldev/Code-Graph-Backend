import json
from fastapi import APIRouter, Depends, HTTPException
from configmanager.service import ConfigManager
from configmanager.schemas import ConfigModel
from db.models import Config  # Import the Config model

# Create a ConfigManager instance with a database session
from db.session import get_db
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/", response_model=ConfigModel)
def create_config(config: ConfigModel = ConfigManager.get_default_model(), db: Session = Depends(get_db)):
    config_manager = ConfigManager(db)

    config_json = json.dumps(config.dict())

    new_config = Config(name=config.name, config=config_json)  # Store the JSON string in the 'config' column
    new_config.model_config = config.model_config
    new_config.embedding_config = config.embedding_config
    new_config.cluster_config = config.cluster_config
    new_config.default_limit = config.default_limit

    config_manager.save_config(new_config)

    return ConfigModel(
        name=new_config.name,
        model_config=new_config.model_config,
        embedding_config=new_config.embedding_config,
        cluster_config=new_config.cluster_config,
        default_limit=new_config.default_limit,
    )


@router.get("/")
def get_all_configs(db: Session = Depends(get_db)):
    config_manager = ConfigManager(db)

    configs = config_manager.get_all_configs()
    return configs


@router.get("/{id}")
def get_config(id: int, db: Session = Depends(get_db)):
    config_manager = ConfigManager(db)

    config = config_manager.get_config(id)
    if config:
        return config
    else:
        raise HTTPException(status_code=404, detail=f"Config '{id}' does not exist.")


@router.put("/{id}")
def update_config(id: int, config: ConfigModel = ConfigManager.get_default_model(), db: Session = Depends(get_db)):
    config_manager = ConfigManager(db)

    existing_config = config_manager.get_config(id)
    if existing_config:
        config_json = json.dumps(config.dict())

        existing_config.name = config.name
        existing_config.config = config_json  # Update the 'config' JSON string

        existing_config.model_config = config.model_config
        existing_config.embedding_config = config.embedding_config
        existing_config.cluster_config = config.cluster_config
        existing_config.default_limit = config.default_limit

        config_manager.save_config(existing_config)

        return {"message": f"Config '{id}' updated successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Config '{id}' not found.")


@router.delete("/{id}")
def delete_config(id: int, db: Session = Depends(get_db)):
    config_manager = ConfigManager(db)

    config_manager.delete_config(id)
    return {"message": f"Config '{id}' deleted successfully."}


@router.get("/project/{project_id}")
def get_project_config(project_id: int, db: Session = Depends(get_db)):
    config_manager = ConfigManager(db)

    config = config_manager.get_project_config(project_id)
    if config:
        return config
    else:
        raise HTTPException(status_code=404, detail=f"Config for project '{project_id}' does not exist.")
