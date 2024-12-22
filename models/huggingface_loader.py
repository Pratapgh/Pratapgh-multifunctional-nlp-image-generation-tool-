from transformers import pipeline

def load_pipeline(task):
    return pipeline(task)
