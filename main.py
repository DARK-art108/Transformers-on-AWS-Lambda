from sklearn import pipeline
from src.classifier import Classifier

pipeline = Classifier()
def lambda_handler(event, context):
    try:
        return pipeline(event)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": str(e),
        }