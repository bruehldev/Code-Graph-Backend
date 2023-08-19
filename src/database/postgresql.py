import json
from sqlalchemy import create_engine, Column, Integer, String, Text, ARRAY, inspect, MetaData, ForeignKey, Table
from sqlalchemy.types import Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.dialects.postgresql import ARRAY
import logging

logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)

DATABASE_URL = env["DATABASE_URL"]

engine = create_engine(DATABASE_URL)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()
metadata = MetaData()


class DataTable(Base):
    __table__ = Table(
        "default_data_name",
        metadata,
        Column("id", Integer, primary_key=True, index=True),
        Column("sentence", Text),
        Column("segment", String),
        Column("annotation", String),
        Column("position", Integer),
    )


class ReducedEmbeddingsTable(Base):
    __table__ = Table("default_reduced_embededdings", metadata, Column("id", Integer, primary_key=True, index=True), Column("reduced_embeddings", ARRAY(Float)))


def init_table(table_name, table_class):
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        data_table_class = table_class
        data_table_class.__table__.name = table_name  # Update the table name
        data_table_class.__table__.create(bind=engine)
        logger.info(f"Initialized table: {table_name}")
    else:
        logger.info(f"Using table {table_name}")


def get_table_names():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    return table_names


def get_table_info(table_class):
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    table_info = {}

    session = SessionLocal()

    for table_name in table_names:
        table_class.__table__.name = table_name
        row_count = session.query(table_class).count()
        table_info[table_name] = {"table_name": table_name, "row_count": row_count}

    session.close()
    return table_info


def delete_table(table_name, engine=engine):
    Base = declarative_base()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        Base.metadata.drop_all(engine, [table], checkfirst=True)
        logger.info(f"Table '{table_name}' dropped")
    else:
        raise ValueError(f"Table '{table_name}' does not exist")


# Start the database engine
def start_engine():
    engine.connect()


# Disconnect the database engine
def stop_engine():
    engine.dispose()


def get_session():
    return SessionLocal


def get_data(table_name, start, end, table_class):
    table_class.__table__.name = table_name
    session = SessionLocal()
    try:
        data_range = session.query(table_class).slice(start, end).all()
        logger.info(f"Loaded data from database: {table_name}")
        return data_range
    finally:
        session.close()


def get(table_name, table_class, data_id):
    """
    Get data from the specified table by ID.

    :param table_name: Name of the table to fetch data from.
    :param data_id: ID of the data to retrieve.
    :return: Data row with the specified ID.
    """
    table_class.__table__.name = table_name
    session = SessionLocal()
    try:
        data = session.query(table_class).filter_by(id=data_id).first()
        logger.info(f"Loaded data from database: {table_name}")
        if data:
            return data
        else:
            raise ValueError(f"Data with ID {data_id} not found.")
    finally:
        session.close()


# data: sentence, segment, annotation, position
# reduced_embeddings: reduced_embeddings
def create(table_name, table_class, **kwargs):
    # Create an instance of the dynamic table class with the data
    data_table_class = table_class
    data_table_class.__table__.name = table_name  # Update the table name
    plot_instance = data_table_class(**kwargs)
    # Add the instance to the session
    SessionLocal.add(plot_instance)
    SessionLocal.commit()


def update(table_name, table_class, data_id, new_values):
    """
    Update data in the specified table.

    :param data_id: ID of the data to update.
    :param new_values: Dictionary containing the new values for the fields to update.
    """
    table_class.__table__.name = table_name
    session = SessionLocal()
    try:
        data = session.query(table_class).filter_by(id=data_id).first()
        if data:
            for key, value in new_values.items():
                setattr(data, key, value)
            session.commit()
        else:
            raise ValueError(f"Data with ID {data_id} not found.")
    finally:
        session.close()


def delete(table_name, table_class, data_id):
    """
    Delete data from the specified table.

    :param data_id: ID of the data to delete.
    :return: True if data was deleted, False if data was not found.
    """
    table_class.__table__.name = table_name
    session = SessionLocal()
    try:
        data_to_delete = session.query(table_class).filter_by(id=data_id).first()
        if data_to_delete:
            session.delete(data_to_delete)
            session.commit()
            return True
        else:
            return False
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def table_has_entries(table_name, table_class):
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return False

    table_class.__table__.name = table_name
    session = SessionLocal()
    try:
        count = session.query(table_class).count()
        return count > 0
    finally:
        session.close()
