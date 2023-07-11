from fastapi import APIRouter, Depends
import json
from configmanager.service import ConfigManager
from configmanager.schemas import ConfigModel


env = {}

with open("../env.json") as f:
    env = json.load(f)

config_manager = ConfigManager(env["configs"])

router = APIRouter()


@router.post("/", response_model=ConfigModel)
def create_config(config: ConfigModel):
    config_manager.configs[config.name] = config.dict()
    config_manager.save_configs()
    return config


@router.get("/{name}")
def get_config(name: str):
    if name in config_manager.configs:
        return config_manager.configs[name]
    else:
        return {"message": f"Config '{name}' does not exist."}


@router.get("/")
def get_all_configs():
    return config_manager.configs


@router.put("/{name}")
def update_config(name: str, config: ConfigModel):
    if name in config_manager.configs:
        config_manager.configs[name] = config.dict()
        config_manager.save_configs()
        return {"message": f"Config '{name}' updated successfully."}
    else:
        return {"message": f"Config '{name}' does not exist."}


@router.delete("/{name}")
def delete_config(name: str):
    if name in config_manager.configs:
        del config_manager.configs[name]
        config_manager.save_configs()
        return {"message": f"Config '{name}' deleted successfully."}
    else:
        return {"message": f"Config '{name}' does not exist."}
