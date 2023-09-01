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
    table_has_entries,
)
from data.utils import get_path_key

from data.schemas import DataResponse, Dataset_names, Experimental_dataset_names
from segments.schemas import SegmentTable, DataSegmentResponse, SegmentEntry, SegmentData

router = APIRouter()


@router.get("/extract")
def extract_segments_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100, all: bool = False, return_data: bool = False) -> SegmentTable:
    table_name = get_path_key("segments", dataset_name)
    segment_table = get_segment_table(table_name)
    segments = []
    extracted_num = 0
    if all:
        extracted_num = extract_segments(dataset_name)
        if return_data:
            segments = get_all_db(segment_table, as_dict=True)
    if table_has_entries(segment_table):
        length = get_table_length(segment_table)
        extracted_num = extract_segments(dataset_name, start=(page - 1) * page_size, end=page * page_size)
        # Shift start and end by the length of the table
        start = length
        end = start + page_size
        if return_data:
            segments = get_all_db(segment_table, start=start, end=end, as_dict=True)
    else:
        extracted_num = extract_segments(dataset_name, start=(page - 1) * page_size, end=page * page_size)
        if return_data:
            segments = get_all_db(segment_table, 0, page_size, as_dict=True)
        page = 1
    return {"data": segments, "length": extracted_num, "page": page, "page_size": page_size}


@router.get("/")
def get_segments_route(dataset_name: Dataset_names, page: int = 1, page_size: int = 100, all: bool = False) -> SegmentTable:
    if all:
        segments = get_segments(dataset_name)
        return {"data": segments, "length": len(segments)}
    else:
        segments = get_segments(dataset_name, start=(page - 1) * page_size, end=page * page_size)
        return {"data": segments, "length": len(segments), "page": page, "page_size": page_size}


@router.get("/{id}")
def get_segment_route(
    dataset_name: Experimental_dataset_names,
    id: int,
) -> DataSegmentResponse:
    table_name = get_path_key("segments", dataset_name)
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
    table_name = get_path_key("segments", dataset_name)
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
    table_name = get_path_key("segments", dataset_name)
    segment_table = get_segment_table(table_name)

    try:
        deleted = delete_in_db(segment_table, id)

        return {"id": id, "deleted": deleted}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")


@router.put("/{id}")
def update_segment_route(
    dataset_name: Experimental_dataset_names,
    id: int = 0,
    segment_data: SegmentData = {"sentence": "test", "segment": "test", "annotation": "test", "position": 0},
) -> DataSegmentResponse:
    table_name = get_path_key("segments", dataset_name)
    segment_table = get_segment_table(table_name)

    response = None
    try:
        response = update_in_db(segment_table, id, segment_data.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

    return {"data": response}
