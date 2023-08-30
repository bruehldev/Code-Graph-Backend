from fastapi import APIRouter, Depends
from plot.service import get_plot
from plot.file_operations import extract_plot
from data.schemas import Experimental_dataset_names
from models.schemas import Model_names
from plot.schemas import PlotData, PlotEntry, PlotTable, DataPlotResponse

router = APIRouter()


@router.get("/")
def get_plot_endpoint(
    dataset_name: Experimental_dataset_names,
    model_names: Model_names,
    page: int = 1,
    page_size: int = 100,
) -> PlotTable:
    segments = get_plot(dataset_name, model_names, start=(page - 1) * page_size, end=page * page_size)
    return {"data": segments, "page": page, "page_size": page_size, "length": len(segments)}


# extract plot route
@router.get("/extract")
def extract_plot_endpoint(
    dataset_name: Experimental_dataset_names,
    model_names: Model_names,
):
    extract_plot(dataset_name, model_names)
    return {"message": "Plot data extracted successfully"}
