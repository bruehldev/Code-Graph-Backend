from pydantic import BaseModel
from typing import Union


class DeleteResponse(BaseModel):
    id: int
    deleted: bool
