from transformers import pipeline


def answer_question(question, context):
    try:
        # Load the Question Answering pipeline with a pre-trained model
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

        # Prepare the input format for the pipeline
        answer = qa_pipeline(question=question, context=context)

        return answer['answer']

    except ValueError as ve:
        raise ValueError("Invalid input for question-answering task") from ve

    except Exception as e:
        raise RuntimeError("An error occurred during question answering") from e
