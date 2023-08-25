from fastapi import APIRouter, HTTPException, status
from typing import List
from pydantic import BaseModel
from segements.service import get_segments, extract_segments
from database.schemas import Data, DataTableResponse
from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names
from database.postgresql import (
    get_data as get_all_db,
    get as get_in_db,
    create as create_in_db,
    update as update_in_db,
    delete as delete_in_db,
    get_segment_table,
)
from data.utils import get_path_key
from embeddings.service import delete_embedding

router = APIRouter()


class DataTableResponse(BaseModel):
    id: int
    sentence: str
    segment: str
    annotation: str
    position: int


@router.get("/extract")
def extract_segments_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100, export_to_file: bool = False):
    extract_segments(dataset_name, start=(page - 1) * page_size, end=page * page_size, export_to_file=export_to_file)
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
    segment_table = get_segment_table(table_name)

    data_range = get_all_db(segment_table, (page - 1) * page_size, page * page_size, True)
    return [row.__dict__ for row in data_range]


@router.get("/{id}", response_model=DataTableResponse)
def get_data_route(
    dataset_name: Experimental_dataset_names,
    id: int,
):
    table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(table_name)

    data = None
    try:
        data = get_in_db(segment_table, id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return data


@router.post("/", response_model=Data)
def insert_data_route(
    dataset_name: Experimental_dataset_names,
    data: Data = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
):
    table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(table_name)

    create_in_db(segment_table, sentence=data.sentence, segment=data.sentence, annotation=data.annotation, position=data.position)
    return data


@router.delete("/{id}")
def delete_data_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
):
    table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(table_name)

    try:
        deleted = delete_in_db(segment_table, id)
        if deleted:
            print("TODO: delete embedding")
            # delete_embedding(id, dataset_name)
        return {"id": id, "deleted": deleted}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")


@router.put("/{id}", response_model=Data)
def update_data_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
    data: Data = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
):
    table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(table_name)

    update_in_db(segment_table, id, data.dict())
    return data
