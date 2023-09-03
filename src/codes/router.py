from fastapi import APIRouter, HTTPException, status
from typing import List, Union, Optional
from pydantic import BaseModel
from codes.service import get_codes, extract_codes, get_top_level_codes, get_leaf_codes, build_category_tree
from database.schemas import Data, DataTableResponse, DataRes
from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names
from database.postgresql import (
    get_data as get_all_db,
    get as get_in_db,
    create as create_in_db,
    update as update_in_db,
    delete as delete_in_db,
    get_code_table,
    has_circular_reference,
    delete_table,
    table_has_entries,
    get_last_id,
    update_or_create,
    get_session,
)
from data.utils import get_path_key

router = APIRouter()


class DataTableResponse(BaseModel):
    id: Optional[int]
    code: str
    top_level_code_id: Union[int, None]


class DataRes(BaseModel):
    code: str
    top_level_code_id: Union[int, None]


@router.get("/extract")
def extract_codes_route(dataset_name: Dataset_names):
    if not table_has_entries(get_code_table(get_path_key("code", dataset_name))):
        extract_codes(dataset_name)
        return {"message": "Codes extracted successfully"}
    else:
        raise HTTPException(status_code=400, detail="Codes already extracted")


@router.get("/")
def get_codes_route(
    dataset_name: Dataset_names,
):
    codes = get_codes(dataset_name)
    return {"codes": codes}


@router.get("/roots")
def get_top_level_codes_route(dataset_name: Dataset_names):
    codes = get_top_level_codes(dataset_name)
    return {"codes": codes}


@router.get("/leaves")
def get_leaf_codes_route(
    dataset_name: Dataset_names,
):
    codes = get_leaf_codes(dataset_name)
    return {"codes": codes}


@router.get("/tree")
def get_code_tree(dataset_name: Dataset_names):
    codes = get_codes(dataset_name)
    codes = build_category_tree(codes)
    return {"codes": codes}


@router.get("/{id}", response_model=DataTableResponse)
def get_code_route(
    dataset_name: Experimental_dataset_names,
    id: int,
):
    table_name = get_path_key("code", dataset_name)
    code_table = get_code_table(table_name)

    data = None
    try:
        data = get_in_db(code_table, id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return data


@router.post("/")
def insert_code_route(
    dataset_name: Experimental_dataset_names,
    data: DataTableResponse = {"code": "test", "top_level_code_id": 24},
):
    table_name = get_path_key("code", dataset_name)
    code_table = get_code_table(table_name)
    last_id = get_last_id(code_table)
    print(last_id)
    session = get_session()

    update_or_create(session, code_table, data_id=last_id + 1, code=data.code, top_level_code_id=data.top_level_code_id)
    return data


@router.delete("/{id}")
def delete_code_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
):
    table_name = get_path_key("code", dataset_name)
    code_table = get_code_table(table_name)

    try:
        deleted = delete_in_db(code_table, id)
        return {"id": id, "deleted": deleted}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")


@router.put("/{id}")
def update_code_route(dataset_name: Experimental_dataset_names, id: int, data: DataRes = {"code": "director", "top_level_code_id": 2}):
    table_name = get_path_key("code", dataset_name)
    code_table = get_code_table(table_name)
    # if not has_circular_reference(code_table, data.top_level_code_id):
    #    print("NOOOOOOO Circular reference detected")

    update_in_db(code_table, id, data.dict())
    return data
