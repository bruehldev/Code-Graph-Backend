from typing import List
from pydantic import BaseModel


class EmbeddingEntry(BaseModel):
    id: int
    embedding: List[float]


class DataEmbeddingResponse(BaseModel):
    data: EmbeddingEntry


class EmbeddingTable(BaseModel):
    length: int
    page: int
    page_size: int
    reduce_length: int
    data: List[EmbeddingEntry]


class EmbeddingData(BaseModel):
    embedding: List[float]
