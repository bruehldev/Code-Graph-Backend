from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ValidationError

class BertArgs(BaseModel):
    pretrained_model_name_or_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english"

class BertModel(BaseModel):
    args: BertArgs = Field(BertArgs())
    model_name: Literal['bert'] = 'bert'

class UmapArgs(BaseModel):
    n_neighbors: int = 15
    n_components: int = 2
    metric: str = "cosine"
    random_state: int = 42
    n_jobs: int = 1
class UmapModel(BaseModel):
    args: UmapArgs = Field(UmapArgs())
    model_name: Literal['umap'] = 'umap'

class DynamicUmapArgs(BaseModel):
    n_neighbors: int = 15

class DynamicUmapModel(BaseModel):
    args: DynamicUmapArgs = Field(DynamicUmapArgs())
    model_name: Literal['dynamic_umap'] = "dynamic_umap"

class HDBScanArgs(BaseModel):
    min_cluster_size: int = 2
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
class HDBScanModel(BaseModel):
    args: HDBScanArgs = Field(HDBScanArgs())
    model_name: Literal["hdbscan"] = "hdbscan"

class ReductionConfig(BaseModel):
    args: UmapArgs = Field(UmapArgs())
    model_name: str = "umap"

class ConfigModel(BaseModel):
    name: str = "default"
    embedding_config: Union[BertModel] = Field(BertModel())
    reduction_config: Union[UmapModel, DynamicUmapModel] = Field(UmapModel(), discriminator='model_name')
    cluster_config: Union[HDBScanModel] = Field(HDBScanModel())
    default_limit: Optional[int] = None
    model_type: Literal["static", "dynamic"] = "static"


# stuff only here for the program to not f up
class ClusterConfig(BaseModel):
    args: HDBScanArgs = HDBScanArgs()
    model_name: str = "hdbscan"

class BertModelArgs(BaseModel):
    pretrained_model_name_or_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english"

class EmbeddingConfig(BaseModel):
    args: BertModelArgs = BertModelArgs()
    model_name: str = "bert"




"""class BertModelArgs(BaseModel):
    pretrained_model_name_or_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english"


class EmbeddingConfig(BaseModel):
    args: BertModelArgs = BertModelArgs()
    model_name: str = "bert"

class DynamicUmapArgs(BaseModel):
    abc: int = 15


class UmapArgs(BaseModel):
    n_neighbors: int = 15
    n_components: int = 2
    metric: str = "cosine"
    random_state: int = 42
    n_jobs: int = 1


class ReductionConfig(BaseModel):
    args: UmapArgs = UmapArgs()
    model_name: str = "umap"

class DynamicUmapConfig(BaseModel):
    args: DynamicUmapArgs = DynamicUmapArgs()
    model_name: str = "dynamic_umap"

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
    reduction_config: Union[ReductionConfig, DynamicUmapConfig] # = ReductionConfig()
    cluster_config: ClusterConfig = ClusterConfig()
    default_limit: Optional[int] = None
    model_type: Literal["static", "dynamic"] = "static"

    @root_validator(pre=True)
    def set_reduction_config_model(cls, values):
        if len(values) == 0:
            values['reduction_config'] = DynamicUmapConfig()
            return values
        print(values)
        reduction = values.get('reduction_config')
        print(reduction)
        model_name = reduction.get('model_name')
        if model_name == 'dynamic_umap':
            values['reduction_config'] = DynamicUmapConfig(**reduction)
        else:
            values['reduction_config'] = ReductionConfig(**reduction)
        print(values)
        return values"""

# umap_config source: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
# hdbscan_config source: https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/hdbscan_.py
