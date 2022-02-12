import pytest
from src import config


@pytest.fixture
def requests():
    return {
        "texts": ["food was great", "food was bad", "i am going out for food"],
        "model_name": config.MODEL_NAME,
        "tokenizer_name": config.TOKENIZER_NAME
    }


@pytest.fixture
def response():
    return {
        'predictions': [{
            'label': 'POSITIVE',
            'score': 0.97
        }, {
            'label': 'NEGATIVE',
            'score': 0.95
        }, {
            'label': 'NEUTRAL',
            'score': 0.69
        }]
    }
