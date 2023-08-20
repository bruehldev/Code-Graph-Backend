from fastapi import APIRouter, Depends
from typing import List
from pydantic import BaseModel

from database.postgresql import get_data, get, create, update, delete, table_has_entries, delete_table, get_table_info
from data.service import get_path_key, DataTable
from database.schemas import Data, DataTableResponse

from data.schemas import Dataset_names, Experimental_dataset_names

router = APIRouter()


@router.get("/tables")
def get_table_names_route():
    return get_table_info(DataTable)


# for table info
@router.get("/{dataset_name}/info")
def get_table_info_route(dataset_name: Experimental_dataset_names):
    return get_table_info(DataTable)


@router.get("/{dataset_name}/has-entries")
def table_has_entries_route(dataset_name: Dataset_names):
    table_name = get_path_key("data", dataset_name)

    has_entries = table_has_entries(table_name, DataTable)
    return {"has_entries": has_entries}


@router.delete("/{dataset_name}")
def delete_table_route(
    dataset_name: Experimental_dataset_names,
):
    table_name = get_path_key("data", dataset_name)

    return {"deleted": delete_table(table_name)}
