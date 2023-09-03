import json
import logging

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
    func,
    exists,
    CheckConstraint,
    insert as insert_sql,
    delete as delete_sql,
    update as update_sql,
    select as select_sql,
)
from sqlalchemy.types import Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, registry, aliased
from sqlalchemy.dialects.postgresql import ARRAY
from collections import defaultdict


env = {}
with open("../env.json") as f:
    env = json.load(f)

DATABASE_URL = env["DATABASE_URL"]

logger = logging.getLogger(__name__)
engine = create_engine(DATABASE_URL)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
mapper_registry = registry()
metadata = MetaData()


### Table definitions ###
def get_segment_table(table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, index=True),
        Column("sentence", Text),
        Column("segment", String),
        Column("annotation", String),
        Column("position", Integer),
        #Column("code_id", Integer, ForeignKey(f"{code_table_name}.id", ondelete="SET NULL")
        extend_existing=True,
    )


# Todo on update calculate reduced embeddings (onupdate)
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

def get_code_table(table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, index=True),
        Column("code", String),
        Column("top_level_code_id", Integer, ForeignKey(f"{table_name}.id", ondelete="SET NULL")),
        extend_existing=True,
    )

class SegmentsTable:
    pass


class ReducedEmbeddingsTable:
    pass


class ClustersTable:
    pass

class CodeTable:
    pass


### Table operations ###
# Todo add class for parent and child table
def init_table(table_name, table_class, parent_table_class=None):
    inspector = inspect(engine)
    logger.info(f"Initialized table: {table_name}")
    if table_name not in inspector.get_table_names():
        # remove index from relationship if it exists
        table_class.indexes.clear()

        # set relationships
        if parent_table_class is not None and hasattr(parent_table_class, "name"):
            mapper_registry.map_imperatively(
                SegmentsTable, parent_table_class, properties={"reduced_embeddings": relationship(table_class, cascade="all,delete")}
            )
            mapper_registry.map_imperatively(
                ReducedEmbeddingsTable, table_class, properties={"segment": relationship(parent_table_class, cascade="all,delete")}
            )
        if "code" in table_name:
             check_constraint = CheckConstraint("id <> top_level_code_id", name="no_circular_reference")
             table_class.append_constraint(check_constraint)
        # create table
        table_class.create(bind=engine)
        logger.info(f"Initialized table: {table_name}")
    else:
        logger.info(f"Using table {table_name}")


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
        elif "code" in table_name:
            table_class = get_code_table(table_name)
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


### Query functions ###
def get_data(table_class, start, end, as_dict=True):
    session = SessionLocal()
    try:
        query = session.query(table_class).slice(start, end)
        if as_dict:
            return [row._asdict() for row in query.all()]
        return query
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get(table_class: Table, id):
    stmt = table_class.select().where(table_class.c.id == id)
    session = SessionLocal()
    try:
        data = session.execute(stmt).first()
        logger.info(f"Loaded data from database: {table_class.name}")
        if data is not None:
            return data._asdict()
        else:
            return None
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# data: sentence, segment, annotation, position
# reduced_embeddings: reduced_embeddings
def create(table_class, **kwargs):
    stmt = insert_sql(table_class).values(**kwargs)
    session = SessionLocal()
    try:
        session.execute(stmt)
        session.commit()
        # logger.info(f"Created data in database: {table_class.name}")
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def update(table_class, data_id, new_values):
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
        logger.info(f"Updated data in database: {table_class.name}")
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def delete(table_class, data_id):
    stmt = delete_sql(table_class).where(table_class.c.id == data_id)
    session = SessionLocal()
    try:
        result = session.execute(stmt)
        deleted_count = result.rowcount
        session.commit()
        logger.info(f"Deleted data from database: {table_class.name}")
        return deleted_count > 0
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def table_has_entries(table_class):
    session = SessionLocal()
    try:
        stmt = select_sql(table_class)
        result = session.execute(stmt)
        row_count = result.rowcount
        return row_count > 0
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()

def get_all_codes(table_class):
    stmt = table_class.select().where(table_class.c.top_level_code_id.is_(None))
    session = SessionLocal()
    try:
        data = session.execute(stmt)
        logger.info(f"Loaded data from database: {table_class.name}")
        if data is not None:
            return [row._asdict() for row in data]
        else:
            return None
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_all_leaf_codes(table_class):
    subquery = (
        table_class.select()
        .add_columns(func.coalesce(table_class.c.top_level_code_id, table_class.c.id).label('leaf_id'))
        .distinct()
        .cte()
        .select()
    )

    stmt = table_class.select().where(~exists().where(table_class.c.id == subquery.c.leaf_id))
    session = SessionLocal()
    try:
        data = session.execute(stmt)
        logger.info(f"Loaded data from database: {table_class.name}")
        if data is not None:
            return [row._asdict() for row in data]
        else:
            return None
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def has_circular_reference(table_class, id):
    session = SessionLocal()
    visited_nodes = set()
    path = []

    def dfs(node_id):
        if node_id in visited_nodes:
            return False

        visited_nodes.add(node_id)
        path.append(node_id)

        parent_alias = aliased(table_class)
        children = (
            session.query(table_class)
            .join(parent_alias, parent_alias.c.id == table_class.c.top_level_code_id)
            .filter(table_class.c.id == node_id)
            .all()
        )

        for child in children:
            child_id = child.id
            if child_id in path or dfs(child_id):
                return True

        path.pop()
        return False

    return dfs(id)