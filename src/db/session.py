import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .base import Base
import os
import json

env = {}
with open("../env.json") as f:
    env = json.load(f)

DATABASE_URL = env["DATABASE_URL"]
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def start_engine():
    engine.connect()


# Disconnect the database engine
def stop_engine():
    engine.dispose()

def init_db():
    Base.metadata.create_all(bind=engine)