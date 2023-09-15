from fastapi import APIRouter, Depends
from plot.service import get_plot
from plot.file_operations import extract_plot
from data.schemas import Experimental_dataset_names, Dataset_names
from models.schemas import Model_names
from plot.schemas import PlotData, PlotEntry, PlotTable, DataPlotResponse
from data.utils import get_path_key
from database.postgresql import (
    get_segment_table,
    plot_search_sentenc,
    get_reduced_embedding_table,
    get_cluster_table,
    plot_search_annotion,
    plot_search_cluster,
    plot_search_segment,
)
from plot.schemas import PlotData, PlotEntry, PlotTable, DataPlotResponse
from db.schemas import DeleteResponse
from project.service import ProjectService
from db.session import get_db
from sqlalchemy.orm import Session, aliased
from fastapi import Depends
from db.models import Cluster, Model, Project, ReducedEmbedding, Segment, Sentence, Code, Embedding

router = APIRouter()


@router.get("/")
def get_plot_endpoint(
    project_id: int,
    all: bool = False,
    page: int = 1,
    page_size: int = 100,
    db: Session = Depends(get_db),
):
    plots = []
    project = ProjectService(project_id, db)
    model_entry = project.get_model_entry("cluster_config")
    return_dict = {}

    ReducedEmbeddingAlias = aliased(ReducedEmbedding)
    EmbeddingAlias = aliased(Embedding)
    SegmentAlias = aliased(Segment)
    SentenceAlias = aliased(Sentence)
    CodeAlias = aliased(Code)
    ProjectAlias = aliased(Project)

    if all:
        query = (
            db.query(
                Cluster,
                ReducedEmbeddingAlias,
                EmbeddingAlias,
                SegmentAlias,
                SentenceAlias,
                CodeAlias,
                ProjectAlias,
            )
            .join(ReducedEmbeddingAlias, Cluster.reduced_embedding_id == ReducedEmbeddingAlias.reduced_embedding_id)
            .join(EmbeddingAlias, ReducedEmbeddingAlias.embedding_id == EmbeddingAlias.embedding_id)
            .join(SegmentAlias, EmbeddingAlias.segment_id == SegmentAlias.segment_id)
            .join(SentenceAlias, SegmentAlias.sentence_id == SentenceAlias.sentence_id)
            .join(CodeAlias, SegmentAlias.code_id == CodeAlias.code_id)
            .join(ProjectAlias, CodeAlias.project_id == ProjectAlias.project_id)
            .filter(Cluster.model_id == model_entry.model_id)
            .all()
        )
        result_dicts = [
            {
                "id": row[3].segment_id,
                "sentence": row[4].text,
                "segment": row[3].text,
                "code": row[5].code_id,
                "reduced_embedding": {"x": row[1].pos_x, "y": row[1].pos_y},
                "cluster": row[0].cluster,
            }
            for row in query
        ]

        print(result_dicts)

        return {"data": result_dicts, "length": len(result_dicts)}
    """
    else: TODO: Implement pagination
        start = (page - 1) * page_size
        end = page * page_size
        segments = get_plot(dataset_name, model_name, start=start, end=end)
        return {"data": segments, "page": page, "page_size": page_size, "length": len(segments)}
    """


@router.get("/sentence/")
def search_segments_route(dataset_name: Dataset_names, model_name: Model_names, query: str, limit: int = 100) -> PlotTable:
    segment_table_name = get_path_key("segments", dataset_name)
    segment_table = get_segment_table(segment_table_name)
    reduced_embedding_table = get_reduced_embedding_table(get_path_key("reduced_embedding", dataset_name, model_name), segment_table_name)
    cluster_table = get_cluster_table(get_path_key("clusters", dataset_name, model_name), segment_table_name)

    plots = plot_search_sentenc(segment_table, reduced_embedding_table, cluster_table, query, as_dict=True, limit=limit)

    return {
        "data": plots,
        "length": len(plots),
        "limit": limit,
    }


@router.get("/annotation/")
def search_annoations_route(dataset_name: Dataset_names, model_name: Model_names, query: str, limit: int = 100) -> PlotTable:
    segment_table_name = get_path_key("segments", dataset_name)
    segment_table = get_segment_table(segment_table_name)
    reduced_embedding_table = get_reduced_embedding_table(get_path_key("reduced_embedding", dataset_name, model_name), segment_table_name)
    cluster_table = get_cluster_table(get_path_key("clusters", dataset_name, model_name), segment_table_name)

    plots = plot_search_annotion(segment_table, reduced_embedding_table, cluster_table, query, as_dict=True, limit=limit)

    return {
        "data": plots,
        "length": len(plots),
        "limit": limit,
    }


@router.get("/cluster/")
def search_clusters_route(dataset_name: Dataset_names, model_name: Model_names, query: int, limit: int = 100) -> PlotTable:
    segment_table_name = get_path_key("segments", dataset_name)
    segment_table = get_segment_table(segment_table_name)
    reduced_embedding_table = get_reduced_embedding_table(get_path_key("reduced_embedding", dataset_name, model_name), segment_table_name)
    cluster_table = get_cluster_table(get_path_key("clusters", dataset_name, model_name), segment_table_name)

    plots = plot_search_cluster(segment_table, reduced_embedding_table, cluster_table, query, as_dict=True, limit=limit)

    return {
        "data": plots,
        "length": len(plots),
        "limit": limit,
    }


@router.get("/segment")
def search_segment_route(dataset_name: Dataset_names, model_name: Model_names, query: str, limit: int = 100) -> PlotTable:
    segment_table_name = get_path_key("segments", dataset_name)
    segment_table = get_segment_table(segment_table_name)
    reduced_embedding_table = get_reduced_embedding_table(get_path_key("reduced_embedding", dataset_name, model_name), segment_table_name)
    cluster_table = get_cluster_table(get_path_key("clusters", dataset_name, model_name), segment_table_name)

    plots = plot_search_segment(segment_table, reduced_embedding_table, cluster_table, query, as_dict=True, limit=limit)

    return {
        "data": plots,
        "length": len(plots),
        "limit": limit,
    }


# extract plot route
@router.get("/exportJSON/")
def extract_plot_endpoint(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
):
    extract_plot(dataset_name, model_name)
    return {"message": "Plot data extracted successfully"}
