from pydantic import BaseModel
from enum import Enum
from typing import List


class Model_names(str, Enum):
    LageCased = "dbmdz%2Fbert-large-cased-finetuned-conll03-english"
    BertModel = "bert-base-uncased"
    BERTopic = "BERTopic"
