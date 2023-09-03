from pydantic import BaseModel
from typing import List, Optional


class PlotData(BaseModel):
    sentence: str
    segment: str
    annotation: str
    position: int
    cluster: int
    reduced_embedding: List[float]


class PlotEntry(BaseModel):
    id: int
    sentence: str
    segment: str
    annotation: str
    position: int
    cluster: Optional[int]
    reduced_embedding: Optional[List[float]]


class PlotTable(BaseModel):
    length: int
    limit: Optional[int]
    page: Optional[int]
    page_size: Optional[int]
    data: List[PlotEntry]


class DataPlotResponse(BaseModel):
    data: PlotEntry