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


def init_data_table(table_name):
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        data_table_class = DataTable
        data_table_class.__table__.name = table_name  # Update the table name
        data_table_class.__table__.create(bind=engine)
        print(f"Initialized table: {table_name}")
    else:
        print(f"Table {table_name} already exists.")


def init_reduced_embeddings_table(table_name):
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        data_table_class = ReducedEmbeddingsTable
        data_table_class.__table__.name = table_name  # Update the table name
        data_table_class.__table__.create(bind=engine)
        print(f"Initialized table: {table_name}")
    else:
        print(f"Table {table_name} already exists.")


def get_table_names():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    return table_names


def get_table_info():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    table_info = {}

    session = SessionLocal()

    for table_name in table_names:
        DataTable.__table__.name = table_name
        row_count = session.query(DataTable).count()
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


def get_all_data(table_name):
    """
    Get all data from the specified table.

    :param table_name: Name of the table to fetch data from.
    :return: List of all data rows from the table.
    """

    DataTable.__table__.name = table_name
    session = SessionLocal()
    try:
        data = session.query(DataTable).all()
        return data
    finally:
        session.close()


def get_data_range(table_name, start, end):
    DataTable.__table__.name = table_name
    session = SessionLocal()
    try:
        data_range = session.query(DataTable).slice(start, end).all()
        logger.info(f"Loaded data from database: {table_name}")
        return data_range
    finally:
        session.close()


def get_reduced_embeddings_range(table_name, start, end):
    ReducedEmbeddingsTable.__table__.name = table_name
    session = SessionLocal()
    try:
        data_range = session.query(ReducedEmbeddingsTable).slice(start, end).all()
        logger.info(f"Loaded data from database: {table_name}")
        return data_range
    finally:
        session.close()


def get_data(table_name, data_id):
    """
    Get data from the specified table by ID.

    :param table_name: Name of the table to fetch data from.
    :param data_id: ID of the data to retrieve.
    :return: Data row with the specified ID.
    """
    DataTable.__table__.name = table_name
    session = SessionLocal()
    try:
        data = session.query(DataTable).filter_by(id=data_id).first()
        logger.info(f"Loaded data from database: {table_name}")
        if data:
            return data
        else:
            raise ValueError(f"Data with ID {data_id} not found.")
    finally:
        session.close()


def insert_data(table_name, sentence, segment, annotation, position):
    # Create an instance of the dynamic table class with the data
    data_table_class = DataTable
    data_table_class.__table__.name = table_name  # Update the table name
    plot_instance = data_table_class(sentence=sentence, segment=segment, annotation=annotation, position=position)
    # Add the instance to the session
    SessionLocal.add(plot_instance)
    SessionLocal.commit()


def insert_reduced_embedding(table_name, reduced_embeddings):
    data_table_class = ReducedEmbeddingsTable
    data_table_class.__table__.name = table_name
    plot_instance = data_table_class(reduced_embeddings=reduced_embeddings)

    SessionLocal.add(plot_instance)
    SessionLocal.commit()


def update_data(table_name, data_id, new_values):
    """
    Update data in the specified table.

    :param data_id: ID of the data to update.
    :param new_values: Dictionary containing the new values for the fields to update.
    """
    DataTable.__table__.name = table_name
    session = SessionLocal()
    try:
        data = session.query(DataTable).filter_by(id=data_id).first()
        if data:
            for key, value in new_values.items():
                setattr(data, key, value)
            session.commit()
        else:
            raise ValueError(f"Data with ID {data_id} not found.")
    finally:
        session.close()


def delete_data(table_name, data_id):
    """
    Delete data from the specified table.

    :param data_id: ID of the data to delete.
    :return: True if data was deleted, False if data was not found.
    """
    DataTable.__table__.name = table_name
    session = SessionLocal()
    try:
        data_to_delete = session.query(DataTable).filter_by(id=data_id).first()
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


def table_has_entries(table_name):
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return False

    DataTable.__table__.name = table_name
    session = SessionLocal()
    try:
        count = session.query(DataTable).count()
        return count > 0
    finally:
        session.close()
