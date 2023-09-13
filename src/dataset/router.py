import json
import shutil
from fastapi import APIRouter, Depends, UploadFile, HTTPException, File, Body
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import session, models
from dataset.schemas import DatasetCreate, DatasetTextOptions
from dataset.service import text_to_json, add_data_to_db

from pprint import pprint

from db.schemas import DeleteResponse

router = APIRouter()


@router.post("/upload")
async def upload_dataset(
    project_id: int,
    dataset_name: str,
    split: str = "\\t",
    sentence_split: str = "\\n\\n",
    word_idx: int = 0,
    label_idx: int = 1,
    label_split: str = "None",
    type: str = "plain",
    file: UploadFile = File(...),
    db: Session = Depends(session.get_db),
):
    # Check if the project exists and belongs to the user
    project = db.query(models.Project).filter(models.Project.project_id == project_id).first()

    options = DatasetTextOptions(
        split=split.encode().decode("unicode_escape"),
        sentence_split=sentence_split.encode().decode("unicode_escape"),
        type=type.encode().decode("unicode_escape"),
        word_idx=word_idx,
        label_idx=label_idx,
        label_split=label_split.encode().decode("unicode_escape"),
    )

    if not project:
        raise HTTPException(status_code=404, detail="Project not found or you don't have permission to access it.")

    file_content = await file.read()
    file_content = file_content.decode("utf-8")

    # 1. By MIME Type
    extension = file.filename.split(".")[-1]
    if extension == "txt":
        file_type = "PlainText"
        temp_dictionary = text_to_json(file_content, options)
    elif extension == "json":
        file_type = "JSON"
        temp_dictionary = json.loads(file_content)
    else:
        return {"error": "Unsupported file type"}

    add_data_to_db(project_id, dataset_name, temp_dictionary, db)

    return {"filename": file.filename, "content_length": len(file_content)}


@router.get("/")
def get_datasets_route(project_id: int, db: Session = Depends(session.get_db)):
    datasets = db.query(models.Dataset).filter(models.Dataset.project_id == project_id).all()
    return datasets


@router.get("/{dataset_id}/")
def get_dataset_route(project_id: int, dataset_id: int, db: Session = Depends(session.get_db)):
    dataset = db.query(models.Dataset).filter(models.Dataset.project_id == project_id and models.Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or you don't have permission to access it.")
    return dataset


@router.put("/{dataset_id}/")
def update_dataset_route(project_id: int, dataset_id: int, dataset_name: str, db: Session = Depends(session.get_db)):
    dataset = db.query(models.Dataset).filter(models.Dataset.project_id == project_id and models.Dataset.dataset_id == dataset_id).first()
    dataset.dataset_name = dataset_name
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.delete("/{dataset_id}/", response_model=DeleteResponse)
def delete_datasets_route(project_id: int, dataset_id: int, db: Session = Depends(session.get_db)):
    dataset = db.query(models.Dataset).filter(models.Dataset.project_id == project_id and models.Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or you don't have permission to access it.")
    db.delete(dataset)
    db.commit()
    return {"id": dataset_id, "deleted": True}
