from transformers import pipeline


def predict_next_word(text):
    try:
        # Load the fill-mask pipeline
        predictor = pipeline("fill-mask", model="bert-base-uncased")

        # Predict the next word
        predicted = predictor(text)
        return predicted

    except ValueError as ve:
        raise ValueError("Invalid input text for next word prediction") from ve

    except Exception as e:
        raise RuntimeError("An error occurred during next word prediction") from e



###Note: please provide input in following format:
###  example: 1. "I love your [MASK]."  also like, 2. what is your [MASK]? etc