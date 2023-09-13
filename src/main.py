import uvicorn
from fastapi import FastAPI

from clusters.service import *
from db.session import engine

from codes.router import router as code_router
from project.router import router as project_router
from dataset.router import router as dataset_router

from db.base import Base
from db.session import engine

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CodeGraph")

app.include_router(project_router, prefix="/project", tags=["project"])
app.include_router(dataset_router, prefix="/project/{project_id}/dataset", tags=["dataset"])
app.include_router(code_router, prefix="/data/{project_id}/codes", tags=["codes"])


@app.get("/")
def read_root():
    return {"status": "online"}


def init_db():
    logger.info(f"Initializing tables: {Base.metadata.tables.keys()}")
    Base.metadata.create_all(bind=engine)


init_db()
