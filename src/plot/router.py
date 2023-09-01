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
from typing import Optional


router = APIRouter()


@router.get("/")
def get_plot_endpoint(
    dataset_name: Experimental_dataset_names,
    model_name: Model_names,
    include_all: Optional[bool] = False,
    page: Optional[int] = 1,
    page_size: Optional[int] = 100,
) -> PlotTable:
    if include_all:
        segments = get_plot(dataset_name, model_name)
        return {"data": segments, "length": len(segments)}
    else:
        start = (page - 1) * page_size
        end = page * page_size
        segments = get_plot(dataset_name, model_name, start=start, end=end)
        return {"data": segments, "page": page, "page_size": page_size, "length": len(segments)}


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
