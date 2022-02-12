from asyncio.log import logger
from distutils.log import INFO
from functools import lru_cache
from transformers import (AutoTokenizer,AutoModelForSequenceClassification,
                         AutoConfig, pipeline)

from src import config,utils
import warnings
import sys
import os
import gc

warnings.filterwarnings("ignore")

logger = utils.create_logger(project_name=config.PREDICTION_TYPE, level=INFO)

class Classifier:
    def __init__(self):
        _ = self.get_sentiment_pipeline(model_name=config.MODEL_NAME, tokenizer_name=config.TOKENIZER_NAME)

    @staticmethod
    @lru_cache(maxsize=config.CACHE_MAXSIZE)
    def get_sentiment_pipeline(model_name: str, tokenizer_name: str) -> pipeline:
        logger.info(f"Loading model {model_name}")
        id2label = config.ID_SENTIMENT_MAPPING[model_name]
        label2id = {v: k for k, v in id2label.items()}

        model_config = AutoConfig.from_pretrained(model_name)
        model_config.label2id = label2id
        model_config.id2label = id2label
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=model_config
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        classification_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return classification_pipeline

    def get_clean_text(str, text: str) -> str:
        return text.strip().lower()

    def __call__(self, request: dict) -> dict:
        texts = [self.get_clean_text(text) for text in request["texts"]]
        model_name = request.get("model_name", config.MODEL_NAME)
        logger.info(f"Predicting sentiment for {len(texts)} using model {model_name}")
        classification_pipeline = self.get_sentiment_pipline(model_name, tokenizer_name)

        predictions = classification_pipeline(texts)
        for i, pred in enumerate(predictions):
            predictions[i]["score"] = round(pred["score"], 2)

        return {
            "predictions": predictions,
        }
