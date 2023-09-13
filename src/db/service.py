import json
import logging
from typing import List
from fastapi import Depends

from sqlalchemy import Table
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, registry, aliased, mapper, Session
from sqlalchemy.schema import DropTable
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.exc import NoSuchTableError

from db.base import Base
from db.models import Project, Dataset, Sentence, Segment, Embedding, ReducedEmbedding, Code, Model, Config
from db.session import get_session, get_engine

engine = get_engine()
metadata = Base.metadata
logger = logging.getLogger(__name__)

table_to_model = {
    "Project": Project,
    "Dataset": Dataset,
    "Sentence": Sentence,
    "Segment": Segment,
    "Embedding": Embedding,
    "ReducedEmbedding": ReducedEmbedding,
    "Code": Code,
    "Model": Model,
    "Config": Config,
}


def init_db():
    logger.info(f"Initializing tables: {Base.metadata.tables.keys()}")
    Base.metadata.create_all(bind=engine)
    return Base.metadata.tables.keys()


def get_table_names():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    return table_names


@compiles(DropTable, "postgresql")
def _postgresql_drop_table_cascade(element, compiler, **kwargs):
    return compiler.visit_drop_table(element) + " CASCADE"


def delete_table(table_name) -> bool:
    try:
        table = Table(table_name, metadata, autoload=True, autoload_with=engine)
        table.drop(engine)
        logger.info(f"Table '{table_name}' dropped")
        return True
    except NoSuchTableError:
        logger.warning(f"Table '{table_name}' does not exist")
        return False


def delete_all_tables():
    table_names = get_table_names()
    results = []

    for table_name in table_names:
        results.append({"name": table_name, "deleted": delete_table(table_name)})

    return results


def get_table_info():
    table_names = get_table_names()
    results = []
    session = get_session()

    for table_name in table_names:
        model_class = table_to_model.get(table_name)
        if model_class:
            count = session.query(model_class).count()
            results.append({"name": table_name, "count": count})

    session.close()

    return results
