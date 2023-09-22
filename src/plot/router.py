from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.orm import Session, aliased

from clusters.router import extract_clusters_endpoint
from dataset.router import upload_dataset
from db.models import Cluster, Code, Dataset, Embedding, Project, ReducedEmbedding, Segment, Sentence
from db.session import get_db
from embeddings.router import extract_embeddings_endpoint
from plot.file_operations import extract_plot
from plot.schemas import PlotTable
from project.router import create_project_route
from reduced_embeddings.router import extract_embeddings_reduced_endpoint

# TODO: dont use the router, move stuff to services
router = APIRouter()


@router.get("/")
def get_plot_endpoint(
    project_id: int,
    all: bool = False,
    page: int = 0,
    page_size: int = 100,
    db: Session = Depends(get_db),
) -> PlotTable:
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
    response: PlotTable = {}
    if all:
        plots = query.all()
    else:
        plots = query.offset(page * page_size).limit(page_size).all()
        response.update({"page": page, "page_size": page_size})
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
    response.update({"data": result_dicts, "length": len(result_dicts)})

    return response
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
) -> PlotTable:
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


@router.get("/code/")
def search_code_route(project_id: int, search_code_id: int, limit: int = 100, db: Session = Depends(get_db)) -> PlotTable:
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
        .where(CodeAlias.code_id == search_code_id)
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


@router.get("/cluster/")
def search_clusters_route(project_id: int, search_cluster_id: int, limit: int = 100, db: Session = Depends(get_db)) -> PlotTable:
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
        .where(search_cluster_id == ClusterAlias.cluster)
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


@router.get("/segment")
def search_segment_route(project_id: int, search_segment_query: str, limit: int = 100, db: Session = Depends(get_db)) -> PlotTable:
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
    search_segment_query = (
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
        .where(SegmentAlias.text_tsv.match(search_segment_query, postgresql_regconfig="english"))
        .join(ReducedEmbeddingAlias, ClusterAlias.reduced_embedding_id == ReducedEmbeddingAlias.reduced_embedding_id)
        .join(EmbeddingAlias, ReducedEmbeddingAlias.embedding_id == EmbeddingAlias.embedding_id)
        .join(SegmentAlias, EmbeddingAlias.segment_id == SegmentAlias.segment_id)
        .join(SentenceAlias, SegmentAlias.sentence_id == SentenceAlias.sentence_id)
        .join(CodeAlias, SegmentAlias.code_id == CodeAlias.code_id)
        .join(ProjectAlias, CodeAlias.project_id == ProjectAlias.project_id)
        .limit(limit)
    )

    plots = search_segment_query.all()
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


# extract plot route
@router.get("/exportToFiles/")
def extract_plot_endpoint(
    project_id: int,
    db: Session = Depends(get_db),
):
    plots = get_plot_endpoint(project_id=project_id, all=True, db=db)
    extract_plot(project_id=project_id, plots=plots["data"])
    return {"message": "Plot data extracted successfully"}
