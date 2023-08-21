from typing import List
from pydantic import BaseModel


class Reduced_Embedding(BaseModel):
    reduced_embeddings: List[float]
