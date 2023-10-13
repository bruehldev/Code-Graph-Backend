from pydantic import BaseModel
from typing import List


class Correction(BaseModel):
    id: int
    pos: List[float]
