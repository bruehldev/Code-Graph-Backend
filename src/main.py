from fastapi import FastAPI

from codes.router import router as code_router
from project.router import router as project_router
from dataset.router import router as dataset_router
from db.router import router as db_router
from configmanager.router import router as config_router

from db.service import init_db


app = FastAPI(title="CodeGraph")

app.include_router(project_router, prefix="/project", tags=["project"])
app.include_router(dataset_router, prefix="/project/{project_id}/dataset", tags=["dataset"])
app.include_router(code_router, prefix="/data/{project_id}/codes", tags=["codes"])
app.include_router(db_router, prefix="/database", tags=["database"])
app.include_router(config_router, prefix="/config", tags=["config"])


@app.get("/")
def read_root():
    return {"status": "online"}


init_db()
