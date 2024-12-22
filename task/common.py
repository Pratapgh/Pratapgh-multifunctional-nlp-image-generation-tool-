from transformers import pipeline

# Example shared function to load a pipeline
def load_pipeline(task, model_name=None):
    """
    Load a pipeline for a specific task.
    :param task: The task type (e.g., "summarization").
    :param model_name: Optional specific model name.
    :return: Hugging Face pipeline object.
    """
    if model_name:
        return pipeline(task, model=model_name)
    return pipeline(task)
