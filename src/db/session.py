from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .base import Base
import json

env = {}
with open("../env.json") as f:
    env = json.load(f)

DATABASE_URL = env["DATABASE_URL"]
engine = create_engine(DATABASE_URL) #, echo=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

