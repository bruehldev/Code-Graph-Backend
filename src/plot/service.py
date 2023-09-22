import json
import logging

from clusters.service import get_clusters
from data.few_nerd import FINE_NER_TAGS_DICT
from reduced_embeddings.service import get_reduced_embeddings
from segments.service import get_segments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = {}
with open("../env.json") as f:
    env = json.load(f)


def get_plot(dataset_name: str, model_name: str, start: int = 0, end: int = None):
    # Important Note: Anti pattern here for testing purpose. To gain performance, we should use only one query
    segments = get_segments(dataset_name, start, end)
    embeddings = get_reduced_embeddings(dataset_name, model_name, start, end)
    clusters = get_clusters(dataset_name, model_name, start, end)

    for segment, embedding, cluster in zip(segments, embeddings, clusters):
        # Check if ids are equal
        assert segment["id"] == embedding["id"] == cluster["id"]

        segment["reduced_embedding"] = embedding["reduced_embedding"]
        segment["cluster"] = cluster["cluster"]
        segment["annotation"] = FINE_NER_TAGS_DICT[segment["annotation"]]

    logger.info(f"Retrieved plot: {dataset_name} / {model_name}")
    return segments
