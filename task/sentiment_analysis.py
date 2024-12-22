from transformers import pipeline

def analyze_sentiment(text):
    try:
        # Load the sentiment analysis pipeline
        sentiment_analysis_pipeline = pipeline("sentiment-analysis")

        # Perform sentiment analysis
        sentiment = sentiment_analysis_pipeline(text)
        return sentiment[0]

    except ValueError as ve:
        raise ValueError("Invalid input text for sentiment analysis") from ve

    except Exception as e:
        raise RuntimeError("An error occurred during sentiment analysis") from e
