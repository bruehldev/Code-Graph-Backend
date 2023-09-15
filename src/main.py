from fastapi import FastAPI

from codes.router import router as code_router
from project.router import router as project_router
from dataset.router import router as dataset_router
from db.router import router as db_router
from configmanager.router import router as config_router
from embeddings.router import router as embeddings_router
from reduced_embeddings.router import router as reduced_embeddings_router

from db.service import init_db


app = FastAPI(title="CodeGraph")

app.include_router(project_router, prefix="/projects", tags=["projects"])
app.include_router(dataset_router, prefix="/projects/{project_id}/datasets", tags=["datasets"])
app.include_router(code_router, prefix="/projects/{project_id}/codes", tags=["codes"])
app.include_router(embeddings_router, prefix="/projects/{project_id}/embeddings", tags=["embeddings"])
app.include_router(reduced_embeddings_router, prefix="/projects/{project_id}/reduced_embeddings", tags=["reduced_embeddings"])

app.include_router(db_router, prefix="/databases", tags=["databases"])
app.include_router(config_router, prefix="/configs", tags=["configs"])


@app.get("/")
def read_root():
    return {"status": "online"}


init_db()
