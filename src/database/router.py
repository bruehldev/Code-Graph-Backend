from fastapi import APIRouter, Depends
from typing import List
from pydantic import BaseModel

from database.postgresql import (
    get_data,
    get,
    create,
    update,
    delete,
    table_has_entries,
    delete_table,
    get_table_info,
    get_table_names,
    SegmentsTable,
    ReducedEmbeddingsTable,
)


router = APIRouter()


@router.get("/tables")
def get_table_names_route():
    return get_table_names()


# for table info
@router.get("/tables/infos")
def get_table_info_route():
    return get_table_info()


@router.delete("/{tables}/{table_name}")
def delete_table_route(
    table_name: str,
):
    return {"deleted": delete_table(table_name)}
