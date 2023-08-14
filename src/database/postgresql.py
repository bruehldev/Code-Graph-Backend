import json
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, create_engine, Table, update
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.orm import scoped_session
from sqlalchemy.dialects.postgresql import TSVECTOR, ARRAY

env = {}
with open("../env.json") as f:
    env = json.load(f)

DATABASE_URL = env["DATABASE_URL"]

engine = create_engine(DATABASE_URL)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()


class Plot(Base):
    __tablename__ = "plot_table"

    id = Column(Integer, primary_key=True, index=True)
    sentence = Column(Text)
    segment = Column(String)
    annotation = Column(String)
    position = Column(Integer)
    embedding = Column(ARRAY(Integer))
    cluster = Column(Integer)
    # search_vector = Column(TSVECTOR)


# Initialize the database
def init_db():
    delete_table("plot_table")
    Base.metadata.create_all(bind=engine)


# Start the database engine
def start_engine():
    init_db()
    engine.connect()


# Disconnect the database engine
def stop_engine():
    engine.dispose()


def get_session():
    return SessionLocal


def get_plot_model():
    return Plot


def delete_table(table_name):
    metadata = Base.metadata
    table = metadata.tables.get(table_name)
    if table is not None:
        table.drop(engine)
    else:
        raise ValueError(f"Table '{table_name}' does not exist")


def insert_data(sentence, segment, annotation, position):
    # Create a PlotModel instance with the data
    plot_instance = Plot(sentence=sentence, segment=segment, annotation=annotation, position=position)

    # Add the instance to the session
    SessionLocal.add(plot_instance)
    SessionLocal.commit()

    # Trigger the search_vector update for the inserted row
    # stmt = update(plot_table).where(plot_table.c.sentence == sentence)
    # session.execute(stmt)
