from fastapi import APIRouter, Depends
from typing import List
from pydantic import BaseModel

from database.postgresql import (
    get_data_range,
    get_data,
    insert_data,
    update_data,
    delete_data,
    table_has_entries,
)
from data.service import get_path_key
from database.schemas import Data, DataTableResponse

from data.schemas import Dataset_names, Experimental_dataset_names

router = APIRouter()


@router.get("/{dataset_name}/has-entries")
def table_has_entries_route(dataset_name: Dataset_names):
    table_name = get_path_key("data", dataset_name)

    has_entries = table_has_entries(table_name)
    return {"has_entries": has_entries}


@router.get("/{dataset_name}", response_model=List[DataTableResponse])
def get_data_range_route(
    dataset_name: Experimental_dataset_names,
    page: int = 1,
    page_size: int = 100,
) -> list:
    table_name = get_path_key("data", dataset_name)

    data_range = get_data_range(table_name, (page - 1) * page_size, page * page_size)
    return [row.__dict__ for row in data_range]


@router.get("/{dataset_name}/{id}", response_model=DataTableResponse)
def get_data_route(
    dataset_name: Experimental_dataset_names,
    id: int,
):
    table_name = get_path_key("data", dataset_name)

    data = get_data(table_name, id)
    return data.__dict__


@router.post("/{dataset_name}", response_model=Data)
def insert_data_route(
    dataset_name: Experimental_dataset_names,
    data: Data = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
):
    table_name = get_path_key("data", dataset_name)

    insert_data(table_name, sentence=data.sentence, segment=data.sentence, annotation=data.annotation, position=data.position)
    return data


@router.delete("/{dataset_name}/{id}")
def delete_data_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
):
    table_name = get_path_key("data", dataset_name)

    return {"id": id, "deleted": delete_data(table_name, id)}


@router.put("/{dataset_name}/{id}", response_model=Data)
def update_data_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
    data: Data = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
):
    table_name = get_path_key("data", dataset_name)

    update_data(table_name, id, data.dict())
    return data
