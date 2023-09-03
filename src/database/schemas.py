from pydantic import BaseModel


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


class PlotCreate(Data):
    pass


class Plot(Data):
    id: int

    class Config:
        orm_mode = True


class DeleteResponse(BaseModel):
    id: int
    deleted: bool
