from fastapi import APIRouter, Depends
from typing import List
from pydantic import BaseModel

from database.postgresql import delete_table, get_table_info, get_table_names, delete_all_tables

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


@router.delete("/all")
def delete_all_tables_route():
    return {"deleted": delete_all_tables()}
