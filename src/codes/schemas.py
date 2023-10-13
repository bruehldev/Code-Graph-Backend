from typing import List

from pydantic import BaseModel


class MergeOperation(BaseModel):
    list_of_codes: List[int]
    new_code_name: str
