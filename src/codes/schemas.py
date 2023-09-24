from pydantic import BaseModel
from typing import List

class MergeOperation(BaseModel):
    list_of_codes: List[int]
    new_code_name: str
