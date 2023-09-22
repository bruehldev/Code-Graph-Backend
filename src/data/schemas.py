from enum import Enum
from typing import List

from pydantic import BaseModel


class DataResponse(BaseModel):
    data: List[str]


class Experimental_dataset_names(str, Enum):
    few_nerd = "few_nerd"
    fetch_20newsgroups = "fetch_20newsgroups"


class Dataset_names(str, Enum):
    few_nerd = "few_nerd"
