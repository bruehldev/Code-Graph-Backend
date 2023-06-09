import json
import os
import json
import logging
import torch
from fastapi import Depends
from typing import List
from bertopic import BERTopic
from transformers import BertModel, BertTokenizerFast
import pandas as pd
from data.service import get_data, get_segments
from configmanager.service import ConfigManager
from tqdm import tqdm
import numpy as np
from urllib.parse import unquote


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = {}
with open("../env.json") as f:
    env = json.load(f)
config_manager = ConfigManager(env["configs"])
config = config_manager.get_default_model()

models = {}


def sanitize_url_string(url_string):
    decoded_string = unquote(url_string)
    return decoded_string.replace("/", "")


class ModelService:
    def __init__(self, dataset_name, model_name="bert-base-uncased"):
        print(f"ModelService: {dataset_name} : {model_name}")
        global models
        self.model = None
        self.model_key = dataset_name + "~" + model_name
        if self.model_key in models:
            self.model = models[self.model_key]

        if self.model:
            logger.info(f"Using model {model_name} from runtime for {dataset_name}")
        else:
            model_path = os.path.join(env["model_path"], dataset_name)
            os.makedirs(model_path, exist_ok=True)

            if model_name:
                if not os.path.exists(os.path.join(model_path, sanitize_url_string(model_name.value))):
                    model = BertModel.from_pretrained(unquote(model_name.value))
                    torch.save(model, os.path.join(model_path, sanitize_url_string(model_name.value)))
                    logger.info(f"Model trained and saved for dataset: {dataset_name}")
                    self.model = model
                else:
                    model = torch.load(os.path.join(model_path, sanitize_url_string(model_name.value)))
                    logger.info(f"Loaded model from file for dataset: {dataset_name}")
                    self.model = model
            elif model_name == "BERTopic":
                if not os.path.exists(os.path.join(model_path, "BERTopic")):
                    model = BERTopic(**config.model_config.dict())
                    topics, probs = model.fit_transform(get_data(dataset_name))
                    model.save(os.path.join(model_path, "BERTopic"))
                    logger.info(f"Model trained and saved for dataset: {dataset_name}")
                    self.model = model
                else:
                    model = BERTopic.load(os.path.join(model_path, "BERTopic"))
                    logger.info(f"Loaded model from file for dataset: {dataset_name}")
                    self.model = model

            models[self.model_key] = self.model

        self.tokenizer = BertTokenizerFast.from_pretrained(unquote(model_name.value))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_name = model_name
        self.model.eval()

    def process_data(self, data):
        def get_char_position(sentence, segment, annotation_position):
            words = sentence.split()
            char_position = sum(len(word) + 1 for word in words[:annotation_position])
            return char_position

        data = pd.DataFrame.from_dict(data)

        # change the amount in config
        default_limit = config.default_limit

        if default_limit:
            data = data[:default_limit]
            logger.info(f"Limiting data to {default_limit} samples")

        sentences = data["sentence"]
        segments = data["segment"]
        start_indexes = data["position"].astype(int)
        embeddings = []
        total_samples = len(data)
        with tqdm(total=total_samples, desc="Processing data") as pbar:
            for sentence, segment, start_index in zip(sentences, segments, start_indexes):
                start_index = start_index - 1  # tried because the other didnt work
                end_index = start_index + len(segment)
                # print(segment)
                sentence_tokenized = self.tokenizer.encode_plus(sentence, return_offsets_mapping=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(sentence_tokenized["input_ids"].to(self.device), attention_mask=sentence_tokenized["attention_mask"].to(self.device))
                token_embeddings = outputs.last_hidden_state
                offsets = sentence_tokenized["offset_mapping"][0].tolist()
                segment_embeddings = []
                for i, offset in enumerate(offsets):
                    start, end = offset
                    if (start >= start_index and end <= end_index) or (start < start_index and end > end_index) or (start < end_index and end > end_index):
                        word = sentence[start:end]
                        # segment_words.append((self.tokenizer.decode(sentence_tokenized['input_ids'][0][i]), word))
                        segment_embeddings.append(token_embeddings[0][i])
                        # print(f"{word} : {start_index} : {end_index}")
                if len(segment_embeddings) == 0:
                    print(f"ERROR: {sentence} : {segment} : {start_index} : {end_index} : {offsets}")

                if segment_embeddings:
                    mean_embeddings = torch.mean(torch.stack(segment_embeddings), dim=0)
                    embeddings.append(mean_embeddings.detach().cpu().numpy().tolist())
                else:
                    # Append np.nan as a placeholder value
                    embeddings.append(np.nan)
                pbar.update(1)
        return embeddings  # return embedding of the sentence

    def get_topic_info(self, dataset_name: str):
        logger.info(f"Getting topic info for dataset: {dataset_name}")
        topic_info = self.model.get_topic_info()
        return topic_info
