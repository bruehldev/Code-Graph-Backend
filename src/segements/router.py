from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from segements.service import get_segments, extract_segments
from database.schemas import Data, DataTableResponse
from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names
from database.postgresql import get_data as get_data_db, get, create, update, delete, SegmentsTable
from data.utils import get_path_key


router = APIRouter()


class DataTableResponse(BaseModel):
    id: int
    sentence: str
    segment: str
    annotation: str
    position: int


@router.get("/extract")
def extract_segments_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100, export_to_file: bool = False):
    extract_segments(dataset_name, page, page_size, export_to_file)
    return {"message": "Segments extracted successfully"}


@router.get("/")
def get_segments_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100):
    segments = get_segments(dataset_name, start=(page - 1) * page_size, end=page * page_size)
    return {"segments": segments}


@router.get("/", response_model=List[DataTableResponse])
def get_data_range_route(
    dataset_name: Experimental_dataset_names,
    page: int = 1,
    page_size: int = 100,
) -> list:
    table_name = get_path_key("data", dataset_name)

    data_range = get_data_db(table_name, (page - 1) * page_size, page * page_size, SegmentsTable)
    return [row.__dict__ for row in data_range]


@router.get("/{id}", response_model=DataTableResponse)
def get_data_route(
    dataset_name: Experimental_dataset_names,
    id: int,
):
    table_name = get_path_key("data", dataset_name)

    data = get(table_name, SegmentsTable, id)
    return data.__dict__


@router.post("/", response_model=Data)
def insert_data_route(
    dataset_name: Experimental_dataset_names,
    data: Data = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
):
    table_name = get_path_key("data", dataset_name)

    create(table_name, SegmentsTable, sentence=data.sentence, segment=data.sentence, annotation=data.annotation, position=data.position)
    return data


@router.delete("/{id}")
def delete_data_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
):
    table_name = get_path_key("data", dataset_name)

    return {"id": id, "deleted": delete(table_name, SegmentsTable, id)}


@router.put("/{id}", response_model=Data)
def update_data_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
    data: Data = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
):
    table_name = get_path_key("data", dataset_name)

    update(table_name, SegmentsTable, id, data.dict())
    return data
