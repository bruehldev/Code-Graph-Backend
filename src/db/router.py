from fastapi import APIRouter, Depends
from typing import List
from pydantic import BaseModel

from db.service import delete_table, get_table_names, delete_all_tables, get_table_info, init_db
from db.session import get_db, get_engine

from sqlalchemy.orm import Session
from sqlalchemy import text, MetaData, Table
from sqlalchemy import MetaData, Table, Column, Integer

router = APIRouter()


@router.get("/tables")
def get_table_names_route():
    return get_table_names()


@router.get("/tables/init")
def init_tables_route():
    return {"initialized": list(init_db())}


@router.delete("/tables")
def delete_all_tables_route():
    return {"deleted": delete_all_tables()}


@router.get("/tables/infos")
def get_table_info_route():
    return get_table_info()


@router.delete("/{table_name}")
async def delete_table_endpoint(table_name: str):
    return {"name": table_name, "deleted": delete_table(table_name)}
