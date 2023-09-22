from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class BertModelArgs(BaseModel):
    pretrained_model_name_or_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english"


class EmbeddingConfig(BaseModel):
    args: BertModelArgs = BertModelArgs()
    model_name: str = "bert"


class UmapArgs(BaseModel):
    n_neighbors: int = 15
    n_components: int = 2
    metric: str = "cosine"
    random_state: int = 42


class ReductionConfig(BaseModel):
    args: UmapArgs = UmapArgs()
    model_name: str = "umap"


class HDBScanArgs(BaseModel):
    min_cluster_size: int = 2
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"


class ClusterConfig(BaseModel):
    args: HDBScanArgs = HDBScanArgs()
    model_name: str = "hdbscan"


class ConfigModel(BaseModel):
    name: str = "default"
    embedding_config: EmbeddingConfig = EmbeddingConfig()
    reduction_config: ReductionConfig = ReductionConfig()
    cluster_config: ClusterConfig = ClusterConfig()
    default_limit: Optional[int] = None
    model_type: Literal["static", "dynamic"] = "static"


# umap_config source: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
# hdbscan_config source: https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/hdbscan_.py
