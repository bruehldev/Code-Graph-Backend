from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, MetaData, Table, text
from sqlalchemy.orm import Session

from db.service import (delete_all_tables, delete_table, get_table_info,
                        get_table_names, init_db)
from db.session import get_db, get_engine

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
def get_table_info_route(db: Session = Depends(get_db)):
    return get_table_info(db)


@router.delete("/{table_name}")
async def delete_table_endpoint(table_name: str):
    return {"name": table_name, "deleted": delete_table(table_name)}
