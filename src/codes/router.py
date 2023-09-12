from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Union, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

from codes.service import get_codes, extract_codes, get_top_level_codes, get_leaf_codes, build_category_tree
from db import session, models

# from database.schemas import Data, DataTableResponse, DataRes
# from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names
from db.schemas import DeleteResponse

##from data.utils import get_path_key

router = APIRouter()


class DataTableResponse(BaseModel):
    id: Optional[int]
    code: str
    top_level_code_id: Union[int, None]


class DataRes(BaseModel):
    code: str
    top_level_code_id: Union[int, None]


# to do pls
"""
@router.get("/extract")
def extract_codes_route(
        db: Session = Depends(session.get_db)
):
    if not table_has_entries(get_code_table(get_path_key("code", dataset_name))):
        extract_codes(dataset_name)
        return {"message": "Codes extracted successfully"}
    else:
        raise HTTPException(status_code=400, detail="Codes already extracted")

"""


@router.get("/")
def get_codes_route(project_id: int, db: Session = Depends(session.get_db)):
    codes = db.query(models.Code).filter(models.Code.project_id == project_id).all()
    build_category_tree(codes)
    return {"data": codes}


@router.get("/roots")
def get_top_level_codes_route(project_id: int, db: Session = Depends(session.get_db)):
    codes = db.query(models.Code).filter(models.Code.project_id == project_id).filter(models.Code.parent_code_id == None).all()
    return {"data": codes}


"""
@router.get("/leaves")
def get_leaf_codes_route(
    dataset_name: Dataset_names,
):
    codes = get_leaf_codes(dataset_name)
    return {"codes": codes}


"""


@router.get("/tree")
def get_code_tree(project_id: int, db: Session = Depends(session.get_db)):
    codes = db.query(models.Code).filter(models.Code.project_id == project_id).all()
    codes = build_category_tree(codes)
    return {"codes": codes}


@router.get("/{id}")
def get_code_route(project_id: int, id: int, db: Session = Depends(session.get_db)):
    data = db.query(models.Code).filter(models.Code.project_id == project_id).filter(models.Code.code_id == id).first()

    return data


@router.delete("/{id}")
def delete_code_route(project_id: int, id: int, db: Session = Depends(session.get_db)):
    try:
        data = db.query(models.Code).filter(models.Code.project_id == project_id).filter(models.Code.code_id == id).first()
        db.delete(data)
        db.commit()
        return {"id": id, "deleted": True}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")

    data = db.query(models.Code).filter(models.Code.project_id == project_id).filter(models.Code.code_id == id).first()
    if not data:
        return {"id": id, "deleted": False}
    db.delete(data)
    db.commit()
    data = db.query(models.Code).filter(models.Code.project_id == project_id).filter(models.Code.code_id == id).first()
    if data:
        return {"id": id, "deleted": False}
    return {"id": id, "deleted": True}


@router.post("/")
def insert_code_route(project_id: int, code_name: str, parent_id: Optional[int] = None, db: Session = Depends(session.get_db)):
    new_code = models.Code(parent_code_id=parent_id, project_id=project_id, text=code_name)
    db.add(new_code)
    db.commit()
    db.refresh(new_code)
    return new_code


@router.put("/{id}")
def update_code_route(project_id: int, code_id: int, code_name: Optional[str] = None, parent_id: Optional[int] = None, db: Session = Depends(session.get_db)):
    data = db.query(models.Code).filter(models.Code.project_id == project_id).filter(models.Code.code_id == code_id).first()
    if code_name:
        data.text = code_name
    data.parent_code_id = parent_id
    db.add(data)
    db.commit()
    db.refresh(data)
    return data
