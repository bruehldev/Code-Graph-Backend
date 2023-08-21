from pydantic import BaseModel


class Cluster(BaseModel):
    cluster: int
