from typing import List
from pydantic import BaseModel


class ReducedEmbeddingData(BaseModel):
    reduced_embedding: List[float]


class ReducedEmbeddingEntry(BaseModel):
    id: int
    reduced_embedding: List[float]


class ReducedEmbeddingTable(BaseModel):
    length: int
    page: int
    page_size: int
    data: List[ReducedEmbeddingEntry]


class DataReducedEmbeddingResponse(BaseModel):
    data: ReducedEmbeddingEntry
