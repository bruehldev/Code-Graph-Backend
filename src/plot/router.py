from fastapi import APIRouter, Depends, UploadFile
from plot.service import get_plot
from plot.file_operations import extract_plot
from data.schemas import Experimental_dataset_names, Dataset_names
from models.schemas import Model_names
from plot.schemas import PlotData, PlotEntry, PlotTable, DataPlotResponse
from data.utils import get_path_key
from database.postgresql import (
    get_segment_table,
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
from db.models import Cluster, Model, Project, ReducedEmbedding, Segment, Sentence, Code, Embedding, Dataset
from embeddings.router import extract_embeddings_endpoint
from reduced_embeddings.router import extract_embeddings_reduced_endpoint
from clusters.router import extract_clusters_endpoint
from dataset.router import upload_dataset
from project.router import create_project_route

# TODO: dont use the router, move stuff to services
router = APIRouter()


@router.get("/")
def get_plot_endpoint(
    project_id: int,
    all: bool = False,
    page: int = 0,
    page_size: int = 100,
    db: Session = Depends(get_db),
):
    print("plot_router")
    extract_embeddings_endpoint(project_id, db=db)
    extract_embeddings_reduced_endpoint(project_id, db=db)
    extract_clusters_endpoint(project_id, db=db)
    plots = []

    ReducedEmbeddingAlias = aliased(ReducedEmbedding)
    EmbeddingAlias = aliased(Embedding)
    SegmentAlias = aliased(Segment)
    SentenceAlias = aliased(Sentence)
    CodeAlias = aliased(Code)
    ProjectAlias = aliased(Project)
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
        .filter(ProjectAlias.project_id == project_id)
        .join(ReducedEmbeddingAlias, Cluster.reduced_embedding_id == ReducedEmbeddingAlias.reduced_embedding_id)
        .join(EmbeddingAlias, ReducedEmbeddingAlias.embedding_id == EmbeddingAlias.embedding_id)
        .join(SegmentAlias, EmbeddingAlias.segment_id == SegmentAlias.segment_id)
        .join(SentenceAlias, SegmentAlias.sentence_id == SentenceAlias.sentence_id)
        .join(CodeAlias, SegmentAlias.code_id == CodeAlias.code_id)
        .join(ProjectAlias, CodeAlias.project_id == ProjectAlias.project_id)
        .limit(page_size)
    )
    if all:
        plots = query.all()

    else:
        plots = query.offset(page * page_size).limit(page_size).all()

    result_dicts = [
        {
            "id": row[3].segment_id,
            "sentence": row[4].text,
            "segment": row[3].text,
            "code": row[5].code_id,
            "reduced_embedding": {"x": row[1].pos_x, "y": row[1].pos_y},
            "cluster": row[0].cluster,
        }
        for row in plots
    ]
    return {"data": result_dicts, "length": len(result_dicts)}
    """
    else: TODO: Implement pagination
        start = (page - 1) * page_size
        end = page * page_size
        segments = get_plot(dataset_name, model_name, start=start, end=end)
        return {"data": segments, "page": page, "page_size": page_size, "length": len(segments)}
    """


@router.get("/test/")
async def setup_test_environment(db: Session = Depends(get_db)):
    file_os = open("dataset/examples/few_nerd_reduced.txt", "rb")

    file = UploadFile(file_os)
    project = create_project_route(project_name="Test", db=db)
    project_id = project.project_id
    await upload_dataset(project_id, dataset_name="few_nerd_reduced", file=file, db=db)
    extract_embeddings_endpoint(project_id, db=db)
    extract_embeddings_reduced_endpoint(project_id, db=db)
    extract_clusters_endpoint(project_id, db=db)
    return {"message": "Test environment setup successfully", "project_id": project_id}


@router.get("/sentence/")
def search_segments_route(
    project_id: int,
    search_query: str,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    plots = []
    ProjectAlias = aliased(Project)
    SentenceAlias = aliased(Sentence)
    SegmentAlias = aliased(Segment)
    DatasetAlias = aliased(Dataset)
    ReducedEmbeddingAlias = aliased(ReducedEmbedding)
    EmbeddingAlias = aliased(Embedding)
    CodeAlias = aliased(Code)
    ClusterAlias = aliased(Cluster)

    datasets = db.query(DatasetAlias).filter(DatasetAlias.project_id == project_id).all()
    dataset_ids = [dataset.dataset_id for dataset in datasets]
    query = (
        db.query(
            ClusterAlias,
            ReducedEmbeddingAlias,
            EmbeddingAlias,
            SegmentAlias,
            SentenceAlias,
            CodeAlias,
            ProjectAlias,
        )
        .filter(ProjectAlias.project_id == project_id)
        .filter(SentenceAlias.dataset_id.in_(dataset_ids))
        .where(SentenceAlias.text_tsv.match(search_query, postgresql_regconfig="english"))
        .join(ReducedEmbeddingAlias, ClusterAlias.reduced_embedding_id == ReducedEmbeddingAlias.reduced_embedding_id)
        .join(EmbeddingAlias, ReducedEmbeddingAlias.embedding_id == EmbeddingAlias.embedding_id)
        .join(SegmentAlias, EmbeddingAlias.segment_id == SegmentAlias.segment_id)
        .join(SentenceAlias, SegmentAlias.sentence_id == SentenceAlias.sentence_id)
        .join(CodeAlias, SegmentAlias.code_id == CodeAlias.code_id)
        .join(ProjectAlias, CodeAlias.project_id == ProjectAlias.project_id)
        .limit(limit)
    )

    plots = query.all()
    result_dicts = [
        {
            "id": row[3].segment_id,
            "sentence": row[4].text,
            "segment": row[3].text,
            "code": row[5].code_id,
            "reduced_embedding": {"x": row[1].pos_x, "y": row[1].pos_y},
            "cluster": row[0].cluster,
        }
        for row in plots
    ]
    return {"data": result_dicts, "length": len(result_dicts), "limit": limit}


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
