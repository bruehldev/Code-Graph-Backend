from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Union, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

from codes.service import extract_codes, build_category_tree
from db import session, models
from data.schemas import DataResponse
from db.schema import DeleteResponse

router = APIRouter()


class DataTableResponse(BaseModel):
    id: Optional[int]
    code: str
    top_level_code_id: Union[int, None]


class DataRes(BaseModel):
    code: str
    top_level_code_id: Union[int, None]


@router.get("/extract")
def extract_codes_route(dataset_name: str, project_id: int, db: Session = Depends(session.get_db)):
    codes = db.query(models.Code).filter(models.Code.project_id == project_id).all()

    if not codes:
        codes = extract_codes(dataset_name)
        stack = [(None, codes)]
        while stack:
            parent_id, code_data = stack.pop()
            for _, code_info in code_data.items():
                new_code = models.Code(parent_code_id=parent_id, project_id=project_id, text=code_info["name"])
                db.add(new_code)
                subcategories = code_info["subcategories"]
                if subcategories:
                    stack.append((code_info["id"], subcategories))
        db.commit()
        return {"message": "Codes extracted successfully"}
    else:
        raise HTTPException(status_code=400, detail="Codes already extracted")


@router.get("/")
def get_codes_route(project_id: int, db: Session = Depends(session.get_db)):
    codes = db.query(models.Code).filter(models.Code.project_id == project_id).all()
    # build_category_tree(codes)
    return {"data": codes}


@router.get("/roots")
def get_top_level_codes_route(project_id: int, db: Session = Depends(session.get_db)):
    codes = db.query(models.Code).filter(models.Code.project_id == project_id).filter(models.Code.parent_code_id == None).all()
    return {"data": codes}


@router.get("/leaves")
def get_leaf_codes_route(project_id: int, db: Session = Depends(session.get_db)):
    subquery = db.query(models.Code.parent_code_id).filter(models.Code.project_id == project_id).distinct()
    subquery_result = subquery.all()
    code_ids = [item[0] for item in subquery_result if item[0] is not None]
    codes = db.query(models.Code).filter(models.Code.project_id == project_id, ~models.Code.code_id.in_(code_ids)).all()
    return {"codes": codes}


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
