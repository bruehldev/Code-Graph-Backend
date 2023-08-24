import json
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    ARRAY,
    inspect,
    MetaData,
    ForeignKey,
    Table,
    text,
    insert as insert_sql,
    delete as delete_sql,
    update as update_sql,
    select as select_sql,
)
from sqlalchemy.types import Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, registry
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
mapper_registry = registry()
metadata = MetaData()


def get_segment_table(table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, index=True),
        Column("sentence", Text),
        Column("segment", String),
        Column("annotation", String),
        Column("position", Integer),
        extend_existing=True,
    )


def get_reduced_embedding_table(table_name, segment_table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, index=True),
        Column("reduced_embeddings", ARRAY(Float)),
        Column("segment_id", Integer, ForeignKey(f"{segment_table_name}.id", ondelete="CASCADE")),
        extend_existing=True,
    )


def get_cluster_table(table_name, segment_table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, index=True),
        Column("cluster", Integer),
        Column("segment_id", Integer, ForeignKey(f"{segment_table_name}.id", ondelete="CASCADE")),
        extend_existing=True,
    )


class SegmentsTable:
    pass


class ReducedEmbeddingsTable:
    pass


def init_table(table_name, table_class, parent_table_class=None):
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        # remove index from relationship if it exists
        drop_index(table_name)

        # set relationships
        if parent_table_class is not None and hasattr(parent_table_class, "name"):
            mapper_registry.map_imperatively(
                SegmentsTable, parent_table_class, properties={"reduced_embeddings": relationship(table_class, cascade="all,delete")}
            )
            mapper_registry.map_imperatively(
                ReducedEmbeddingsTable, table_class, properties={"segment": relationship(parent_table_class, cascade="all,delete")}
            )

        # create table
        table_class.create(bind=engine)
        logger.info(f"Initialized table: {table_name}")
    else:
        logger.info(f"Using table {table_name}")


def drop_index(table_name):
    logger.info(f"Dropping index: {table_name}")

    # log all existing first
    inspector = inspect(engine)
    for name in inspector.get_table_names():
        for index in inspector.get_indexes(name):
            logger.info("Existing index: ", index)

    session = SessionLocal()
    stmt = text(f"DROP INDEX IF EXISTS {table_name} CASCADE")
    session.execute(stmt)
    session.commit()

    ix_name = f"ix_{table_name}"
    stmt = text(f"DROP INDEX IF EXISTS {ix_name} CASCADE")
    session.execute(stmt)
    session.commit()

    session.close()


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
        # Little bit hacky, but it works :D
        if "data" in table_name:
            table_class = get_segment_table(table_name)
            row_count = session.query(table_class).count()
        elif "reduced_embedding" in table_name:
            table_class = get_reduced_embedding_table(table_name, "data")
            row_count = session.query(table_class).count()
        elif "clusters" in table_name:
            table_class = get_cluster_table(table_name, "data")
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


def table_has_entries(table_name, table_class):
    session = SessionLocal()
    try:
        stmt = select_sql(table_class)
        result = session.execute(stmt)
        row_count = result.rowcount
        return row_count > 0
    finally:
        session.close()


def get_data(table_name, start, end, table_class, as_dict=True):
    session = SessionLocal()
    try:
        query = session.query(table_class).slice(start, end)
        if as_dict:
            return [row._asdict() for row in query.all()]
        return query
    finally:
        session.close()


def get(table_name, table_class: Table, id):
    stmt = table_class.select().where(table_class.c.id == id)
    session = SessionLocal()
    try:
        data = session.execute(stmt).first()
        logger.info(f"Loaded data from database: {table_name}")
        if data is not None:
            return data._asdict()
        else:
            return None
    finally:
        session.close()


# data: sentence, segment, annotation, position
# reduced_embeddings: reduced_embeddings
def create(table_name, table_class, **kwargs):
    stmt = insert_sql(table_class).values(**kwargs)
    SessionLocal.execute(stmt)
    SessionLocal.commit()


def update(table_name, table_class, data_id, new_values):
    """
    Update data in the specified table.

    :param data_id: ID of the data to update.
    :param new_values: Dictionary containing the new values for the fields to update.
    """
    stmt = update_sql(table_class).where(table_class.c.id == data_id).values(**new_values)
    session = SessionLocal()
    try:
        session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def delete(table_name, table_class, data_id):
    stmt = delete_sql(table_class).where(table_class.c.id == data_id)
    session = SessionLocal()
    try:
        result = session.execute(stmt)
        deleted_count = result.rowcount
        session.commit()
        logger.info(f"Deleted data from database: {table_name}")
        return deleted_count > 0
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def table_has_entries(table_name, table_class):
    session = SessionLocal()
    try:
        stmt = select_sql(table_class)
        result = session.execute(stmt)
        row_count = result.rowcount
        return row_count > 0
    finally:
        session.close()
