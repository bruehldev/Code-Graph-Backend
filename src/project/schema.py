from pydantic import BaseModel

from typing import List


class ProjectEntry(BaseModel):
    project_name: str
    project_id: int
    config_id: int


class ProjectData(BaseModel):
    data: ProjectEntry


class ProjectsData(BaseModel):
    data: List[ProjectEntry]
    length: int
