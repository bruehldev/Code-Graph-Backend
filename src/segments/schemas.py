from pydantic import BaseModel
from typing import List


class SegmentData(BaseModel):
    sentence: str
    segment: str
    annotation: str
    position: int


class SegmentEntry(BaseModel):
    id: int
    sentence: str
    segment: str
    annotation: str
    position: int


class SegmentTable(BaseModel):
    length: int
    page: int
    page_size: int
    data: List[SegmentEntry]


class DataSegmentResponse(BaseModel):
    data: SegmentEntry
