from transformers import pipeline

def summarize_text(text):
    try:
        # Load the summarization pipeline
        summarization_pipeline = pipeline("summarization")

        # Perform text summarization
        summary = summarization_pipeline(text, max_length=80, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    except ValueError as ve:
        raise ValueError("Invalid input text for summarization") from ve

    except Exception as e:
        raise RuntimeError("An error occurred during text summarization") from e

# Test the summarization function
text = """
Hugging Face is a company that provides machine learning tools for NLP tasks. 
It has become one of the most well-known organizations in the NLP space due to its popular open-source libraries and models, 
such as BERT, GPT-2, and many others. The company has been building cutting-edge tools for transforming and 
fine-tuning models to solve real-world problems in NLP.
"""

summary = summarize_text(text)
print("Summary:", summary)
