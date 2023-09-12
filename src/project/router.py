from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db.schemas import DeleteResponse

from db.models import Project
from db.session import get_db

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
