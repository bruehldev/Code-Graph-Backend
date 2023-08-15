import json
from sqlalchemy import create_engine, Column, Integer, String, Text, ARRAY
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


class DataTable(Base):
    __tablename__ = "default_data_name"
    id = Column(Integer, primary_key=True, index=True)
    sentence = Column(Text)
    segment = Column(String)
    annotation = Column(String)
    position = Column(Integer)


def init_db(table_name):
    # delete_table("plot_table")
    DataTable.__table__.name = table_name
    table = DataTable()

    table.metadata.create_all(bind=engine)


def delete_table(table_name):
    metadata = Base.metadata
    table = metadata.tables.get(table_name)
    if table is not None:
        table.drop(engine)
        logger.info(f"Table '{table_name}' deleted")
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
    DataTable.__table__.name = table_name
    plot_instance = DataTable(__tablename__=table_name, sentence=sentence, segment=segment, annotation=annotation, position=position)
    # Add the instance to the session
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
    DataTable.__table__.name = table_name
    session = SessionLocal()
    try:
        count = session.query(DataTable).count()
        return count > 0
    finally:
        session.close()
