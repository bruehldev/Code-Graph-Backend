from fastapi import APIRouter, HTTPException, status
from typing import List
from segments.service import get_segments, extract_segments
from database.schemas import DeleteResponse
from database.postgresql import (
    get_data as get_all_db,
    get as get_in_db,
    create as create_in_db,
    update as update_in_db,
    delete as delete_in_db,
    get_segment_table,
    get_table_length,
)
from data.utils import get_path_key

from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names
from segments.schemas import SegmentTable, DataSegmentResponse, SegmentEntry, SegmentData
from embeddings.service import delete_embedding

router = APIRouter()


@router.get("/extract")
def extract_segments_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100, export_to_file: bool = False, all: bool = False) -> SegmentTable:
    table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(table_name)
    length = get_table_length(segment_table)
    extract_segments(dataset_name, start=(page - 1) * page_size, end=page * page_size, export_to_file=export_to_file)
    start = length
    end = start + page_size
    segments = get_all_db(segment_table, start=start, end=end, as_dict=True)
    return {"data": segments, "length": len(segments), "page": page, "page_size": page_size}


@router.get("/")
def get_segments_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100) -> SegmentTable:
    segments = get_segments(dataset_name, start=(page - 1) * page_size, end=page * page_size)

    return {"data": segments, "length": len(segments), "page": page, "page_size": page_size}


@router.get("/{id}")
def get_segment_route(
    dataset_name: Experimental_dataset_names,
    id: int,
) -> DataSegmentResponse:
    table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(table_name)

    data = None
    try:
        data = get_in_db(segment_table, id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")
    return {"data": data}


@router.post("/")
def insert_segment_route(
    dataset_name: Experimental_dataset_names,
    segment_data: SegmentData = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
) -> DataSegmentResponse:
    table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(table_name)

    response = None
    try:
        response = create_in_db(
            segment_table, sentence=segment_data.sentence, segment=segment_data.sentence, annotation=segment_data.annotation, position=segment_data.position
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if response is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create data")

    return {"data": response}


@router.delete("/{id}", response_model=DeleteResponse)
def delete_segment_route(
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


@router.put("/{id}")
def update_segment_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
    segment_data: SegmentData = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
) -> DataSegmentResponse:
    table_name = get_path_key("data", dataset_name)
    segment_table = get_segment_table(table_name)

    response = None
    try:
        response = update_in_db(segment_table, id, segment_data.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

    return {"data": response}
