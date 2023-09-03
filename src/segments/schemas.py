from pydantic import BaseModel
from typing import List, Optional


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
    limit: Optional[int]
    page: Optional[int]
    page_size: Optional[int]
    data: List[SegmentEntry]


class DataSegmentResponse(BaseModel):
    data: SegmentEntry