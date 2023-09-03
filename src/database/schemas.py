from pydantic import BaseModel
from typing import Union


class Data(BaseModel):
    sentence: str
    segment: str
    annotation: str
    position: int


class SegmentTableResponse(BaseModel):
    id: int
    sentence: str
    segment: str
    annotation: str
    position: int

class DataTableResponse(BaseModel):
    id: int
    code: str
    top_level_code_id: Union[int, None]

class DataRes(BaseModel):
    code: str
    top_level_code_id: Union[int, None]


class PlotCreate(Data):
    pass


class Plot(Data):
    id: int

    class Config:
        orm_mode = True


class DeleteResponse(BaseModel):
    id: int
    deleted: bool
