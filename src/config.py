PREDICTION_TYPE = "classification"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
TOKENIZER_NAME = "roberta-base"
ID_SENTIMENT_MAPPING = {
    "cardiffnlp/twitter-roberta-base-sentiment": {
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    "cardiffnlp/twitter-roberta-base-sentiment-large": {
        0: "anger",
        1: "joy",
        2: "optimism",
        3: "sadness",
    }
}
CACHE_MAXSIZE = 4