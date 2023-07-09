from pydantic import BaseModel
from enum import Enum
from typing import List


class Model_names(str, Enum):
    BertModel = "bert-base-uncased"
    BERTopic = "BERTopic"
