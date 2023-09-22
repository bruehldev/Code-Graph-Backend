from typing import Union

from pydantic import BaseModel


class DeleteResponse(BaseModel):
    id: int
    deleted: bool
