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
    Computed,
    text,
)
from sqlalchemy.types import Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, registry, aliased, mapper, Session
from sqlalchemy.dialects.postgresql import ARRAY, insert as insert_dialect, TSVECTOR, BYTEA
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
        Column("sentence_tsv", TSVECTOR, Computed("to_tsvector('english', sentence)")),
        Column("segment", String),
        Column("annotation", String),
        Column("position", Integer),
        # Column("code_id", Integer, ForeignKey(f"{code_table_name}.id", ondelete="SET NULL")
        extend_existing=True,
    )


def get_embedding_table(table_name, segment_table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, ForeignKey(f"{segment_table_name}.id", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True, nullable=False),
        Column("embedding", BYTEA),
        extend_existing=True,
    )


# TODO on update calculate reduced embeddings (onupdate)
def get_reduced_embedding_table(table_name, segment_table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, ForeignKey(f"{segment_table_name}.id", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True, nullable=False),
        Column("reduced_embedding", ARRAY(Float)),
        extend_existing=True,
    )


# TODO on update calculate clusters (onupdate)
def get_cluster_table(table_name, segment_table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, ForeignKey(f"{segment_table_name}.id", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True, nullable=False),
        Column("cluster", Integer),
        extend_existing=True,
    )


def get_code_table(table_name):
    return Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, index=True),
        Column("code", String),
        Column("top_level_code_id", Integer),  # , ForeignKey(f"{table_name}.id", ondelete="SET NULL")
        extend_existing=True,
    )


class SegmentTable:
    pass


class EmbeddingTable:
    pass


class ReducedEmbeddingTable:
    pass


class CodeTable:
    pass


class ClusterTable:
    pass


class MapperFactory:
    def __init__(self, base_class):
        self.base_class = base_class
        self.counter = 1

    def create(self):
        class_name = f"{self.base_class.__name__}{self.counter}"
        new_class = type(class_name, (self.base_class,), {})
        self.counter += 1
        return new_class


### Table operations ###
def init_table(table_name, table_class, parent_table_class=None, cls=None):
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        # remove index from relationship if it exists
        table_class.indexes.clear()

        if parent_table_class is not None and hasattr(parent_table_class, "name") and cls is not None:
            if isinstance(cls, EmbeddingTable):
                mapper_factory = MapperFactory(EmbeddingTable)
                mapper_registry.map_imperatively(
                    mapper_factory.create(), parent_table_class, properties={"embedding": relationship(table_class.name, cascade="all,delete")}
                )
                mapper_registry.map_imperatively(
                    mapper_factory.create(), table_class, properties={"segment": relationship(parent_table_class.name, cascade="all,delete")}
                )
            elif isinstance(cls, ReducedEmbeddingTable):
                mapper_factory = MapperFactory(ReducedEmbeddingTable)
                mapper_registry.map_imperatively(
                    mapper_factory.create(), parent_table_class, properties={"reduced_embedding": relationship(table_class.name, cascade="all,delete")}
                )
                mapper_registry.map_imperatively(
                    mapper_factory.create(), table_class, properties={"segment": relationship(parent_table_class.name, cascade="all,delete")}
                )
            elif isinstance(cls, ClusterTable):
                mapper_factory = MapperFactory(ClusterTable)
                mapper_registry.map_imperatively(
                    mapper_factory.create(), parent_table_class, properties={"cluster": relationship(table_class.name, cascade="all,delete")}
                )
                mapper_registry.map_imperatively(
                    mapper_factory.create(), table_class, properties={"segment": relationship(parent_table_class.name, cascade="all,delete")}
                )
        # if "code" in table_name:
        # check_constraint = CheckConstraint("id <> top_level_code_id", name="no_circular_reference")
        # table_class.append_constraint(check_constraint)
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
        if "segment" in table_name:
            table_class = get_segment_table(table_name)
            row_count = session.query(table_class).count()
        elif "embedding" in table_name:
            table_class = get_embedding_table(table_name, "data")
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


def delete_all_tables(engine=engine):
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    session = SessionLocal()
    results = []
    for table_name in table_names:
        stmt = text(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        results.append(session.execute(stmt))
        logger.info(f"Table '{table_name}' dropped")
    session.commit()
    session.close()

    mapper_registry.dispose()

    return [result.rowcount for result in results]


def get_table_length(table_class):
    session = SessionLocal()
    try:
        query = session.query(table_class)
        return query.count()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# Start the database engine
def start_engine():
    engine.connect()


# Disconnect the database engine
def stop_engine():
    engine.dispose()


def get_session():
    return SessionLocal


### Query functions ###
def get_data(table_class, start=0, end=None, as_dict=True):
    session = SessionLocal()
    try:
        query = session.query(table_class).order_by(table_class.c.id).slice(start, end)
        if as_dict:
            return [row._asdict() for row in query.all()]
        logger.info(f"Loaded data from database: {table_class.name}")
        return query
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get(table_class: Table, id) -> dict:
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
def create(table_class, **kwargs) -> dict:
    stmt = insert_sql(table_class).values(**kwargs)
    session = SessionLocal()
    try:
        result = session.execute(stmt)
        session.commit()
        # logger.info(f"Created data in database: {table_class.name}")
        # return id
        return {"id": result.inserted_primary_key[0], **kwargs}
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# get last id
def get_last_id(table_class):
    session = SessionLocal()
    try:
        query = session.query(table_class).order_by(table_class.c.id.desc()).first()
        return query.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def batch_insert(session: Session, table_class, entries):
    stmt = insert_dialect(table_class).values(entries)
    try:
        session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e


def batch_update_or_create(session: Session, table_class, entries):
    try:
        stmt = insert_dialect(table_class).values(entries)
        stmt = stmt.on_conflict_do_update(index_elements=[table_class.c.id], set_=entries)

        session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e


def update(table_class, data_id, new_values) -> dict:
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
        return {"id": data_id, **new_values}
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def delete(table_class, data_id) -> bool:
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


def plot_search_sentenc(segment_table, reduced_embedding_table, cluster_table, query, as_dict=True, limit=None):
    # Define the columns
    selected_columns = [
        segment_table.c.id,
        segment_table.c.sentence,
        segment_table.c.segment,
        segment_table.c.annotation,
        segment_table.c.position,
    ]

    # Construct the SQL statement
    stmt = select_sql(*selected_columns).where(segment_table.c.sentence_tsv.match(query))

    if table_has_entries(reduced_embedding_table):
        stmt = stmt.join(reduced_embedding_table, segment_table.c.id == reduced_embedding_table.c.id)
        selected_columns.append(reduced_embedding_table.c.reduced_embedding)
    if table_has_entries(cluster_table):
        stmt = stmt.join(cluster_table, segment_table.c.id == cluster_table.c.id)
        selected_columns.append(cluster_table.c.cluster)
    if limit is not None:
        stmt = stmt.limit(limit)

    session = SessionLocal()
    try:
        result = session.execute(stmt)

        if as_dict:
            return [row._asdict() for row in result.fetchall()]
        return result.fetchall()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def plot_search_annotion(segment_table, reduced_embedding_table, cluster_table, query, as_dict=True, limit=None):
    escaped_query = query.replace("/", r"\/")

    # Define the columns you want to select
    selected_columns = [
        segment_table.c.id,
        segment_table.c.sentence,
        segment_table.c.segment,
        segment_table.c.annotation,
        segment_table.c.position,
    ]

    # Construct the SQL statement
    stmt = select_sql(*selected_columns).where(segment_table.c.annotation.match(escaped_query))

    if table_has_entries(reduced_embedding_table):
        stmt = stmt.join(reduced_embedding_table, segment_table.c.id == reduced_embedding_table.c.id)
        selected_columns.append(reduced_embedding_table.c.reduced_embedding)
    if table_has_entries(cluster_table):
        stmt = stmt.join(cluster_table, segment_table.c.id == cluster_table.c.id)
        selected_columns.append(cluster_table.c.cluster)

    if limit is not None:
        stmt = stmt.limit(limit)  # Apply the limit

    session = SessionLocal()
    try:
        result = session.execute(stmt)

        if as_dict:
            return [row._asdict() for row in result.fetchall()]
        return result.fetchall()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def plot_search_cluster(segment_table, reduced_embedding_table, cluster_table, query, as_dict=True, limit=None):
    # Define the columns you want to select
    selected_columns = [
        segment_table.c.id,
        segment_table.c.sentence,
        segment_table.c.segment,
        segment_table.c.annotation,
        segment_table.c.position,
    ]

    # Construct the SQL statement
    stmt = select_sql(*selected_columns).where(cluster_table.c.cluster == query)

    if table_has_entries(reduced_embedding_table):
        stmt = stmt.join(reduced_embedding_table, segment_table.c.id == reduced_embedding_table.c.id)
        selected_columns.append(reduced_embedding_table.c.reduced_embedding)
    if table_has_entries(cluster_table):
        stmt = stmt.join(cluster_table, segment_table.c.id == cluster_table.c.id)
        selected_columns.append(cluster_table.c.cluster)

    if limit is not None:
        stmt = stmt.limit(limit)  # Apply the limit

    session = SessionLocal()
    try:
        result = session.execute(stmt)

        if as_dict:
            return [row._asdict() for row in result.fetchall()]
        return result.fetchall()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def plot_search_segment(segment_table, reduced_embedding_table, cluster_table, query, as_dict=True, limit=None):
    # Define the columns you want to select
    selected_columns = [
        segment_table.c.id,
        segment_table.c.sentence,
        segment_table.c.segment,
        segment_table.c.annotation,
        segment_table.c.position,
    ]

    # Construct the SQL statement
    stmt = select_sql(*selected_columns).where(segment_table.c.segment.match(query))

    if table_has_entries(reduced_embedding_table):
        stmt = stmt.join(reduced_embedding_table, segment_table.c.id == reduced_embedding_table.c.id)
        selected_columns.append(reduced_embedding_table.c.reduced_embedding)
    if table_has_entries(cluster_table):
        stmt = stmt.join(cluster_table, segment_table.c.id == cluster_table.c.id)
        selected_columns.append(cluster_table.c.cluster)

    if limit is not None:
        stmt = stmt.limit(limit)  # Apply the limit

    session = SessionLocal()
    try:
        result = session.execute(stmt)

        if as_dict:
            return [row._asdict() for row in result.fetchall()]
        return result.fetchall()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def update_or_create(session: Session, table_class, data_id, **kwargs):
    try:
        stmt = insert_dialect(table_class).values(id=data_id, **kwargs)
        stmt = stmt.on_conflict_do_update(index_elements=[table_class.c.id], set_=kwargs)
        session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e


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
    subquery = table_class.select().add_columns(func.coalesce(table_class.c.top_level_code_id, table_class.c.id).label("leaf_id")).distinct().cte().select()

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
        children = session.query(table_class).join(parent_alias, parent_alias.c.id == table_class.c.top_level_code_id).filter(table_class.c.id == node_id).all()

        for child in children:
            child_id = child.id
            if child_id in path or dfs(child_id):
                return True

        path.pop()
        return False

    return dfs(id)
