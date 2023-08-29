from pydantic import BaseModel
from typing import List


class ClusterData(BaseModel):
    cluster: int


class ClusterEntry(BaseModel):
    id: int
    cluster: int


class ClusterTable(BaseModel):
    length: int
    page: int
    page_size: int
    data: List[ClusterEntry]


class DataClusterResponse(BaseModel):
    data: ClusterEntry
