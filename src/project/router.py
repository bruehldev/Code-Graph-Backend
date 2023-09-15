from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db.schemas import DeleteResponse

from db.models import Project
from db.session import get_db
from configmanager.service import ConfigManager
from fastapi import HTTPException
from project.service import ProjectService

router = APIRouter()


@router.post("/")
def create_project_route(project_name: str, db: Session = Depends(get_db)):
    new_project = Project(project_name=project_name)
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    return new_project


@router.get("/")
def get_projects_route(db: Session = Depends(get_db)):
    projects = db.query(Project).all()
    return projects


@router.get("/{project_id}/")
def get_project_route(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    return project


@router.put("/{project_id}/")
def update_project_route(project_id: int, project_name: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    project.project_name = project_name
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}/", response_model=DeleteResponse)
def delete_projects_route(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        return {"id": project_id, "deleted": False}
    db.delete(project)
    db.commit()
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if project:
        return {"id": project_id, "deleted": False}
    return {"id": project_id, "deleted": True}


@router.put("/{project_id}/config/{config_id}/")
def set_project_config_route(project_id: int, config_id: int, db: Session = Depends(get_db)):
    project = ProjectService(project_id, db).set_project_config(config_id)
    return project


@router.get("/project/{project_id}")
def get_project_config(project_id: int, db: Session = Depends(get_db)):
    config = ProjectService(project_id, db).get_project_config()
    if config:
        return config
    else:
        raise HTTPException(status_code=404, detail=f"Config for project '{project_id}' does not exist.")
