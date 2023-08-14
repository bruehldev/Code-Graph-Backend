from pydantic import BaseModel


class PlotBase(BaseModel):
    sentence: str
    segment: str
    annotation: int
    position: int
    embedding: dict
    cluster: int


class PlotCreate(PlotBase):
    pass


class Plot(PlotBase):
    id: int

    class Config:
        orm_mode = True
