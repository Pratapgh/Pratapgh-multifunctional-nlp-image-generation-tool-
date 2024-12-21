import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sacrebleu import corpus_bleu
from nltk.tokenize import word_tokenize
from transformers import pipeline
from difflib import SequenceMatcher

# Define mock data for testing each task
test_data = {
    "text_summarization": {
        "input": "Artificial intelligence is the simulation of human intelligence in machines.",
        "expected": "AI is the simulation of human intelligence."
    },
    "sentiment_analysis": {
        "input": ["I am very happy today!", "This is a terrible mistake."],
        "expected": ["POSITIVE", "NEGATIVE"]
    },
    "next_word_prediction": {
        "input": "I love your <mask>.",
        "expected": ["smile"]
    },
    "story_prediction": {
        "input": "Once upon a time, a brave knight embarked on a journey to save the kingdom.",
    },
    "question_answering": {
        "input": {
            "context": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
            "question": "Who developed the theory of relativity?"
        },
        "expected": "Albert Einstein"
    }
}

# Initialize transformers pipelines
sentiment_pipeline = pipeline("sentiment-analysis")
summarization_pipeline = pipeline("summarization")
qa_pipeline = pipeline("question-answering")
mask_filling_pipeline = pipeline("fill-mask")
text_generation_pipeline = pipeline("text-generation")


# Evaluate Sentiment Analysis
def evaluate_sentiment():
    inputs = test_data["sentiment_analysis"]["input"]
    expected = test_data["sentiment_analysis"]["expected"]
    predicted = [sentiment_pipeline(text)[0]["label"] for text in inputs]
    metrics = {
        "accuracy": accuracy_score(expected, predicted),
        "precision": precision_score(expected, predicted, average="weighted"),
        "recall": recall_score(expected, predicted, average="weighted"),
        "f1": f1_score(expected, predicted, average="weighted")
    }
    return metrics


# Evaluate Text Summarization
def evaluate_text_summarization():
    input_text = test_data["text_summarization"]["input"]
    expected_summary = test_data["text_summarization"]["expected"]
    generated_summary = summarization_pipeline(input_text, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
    generated_tokens = word_tokenize(generated_summary)
    expected_tokens = word_tokenize(expected_summary)
    bleu_score = corpus_bleu([generated_tokens], [[expected_tokens]])
    metrics = {"BLEU": bleu_score.score}
    return metrics


# Evaluate Next Word Prediction
def evaluate_next_word_prediction():
    input_text = test_data["next_word_prediction"]["input"]
    expected = test_data["next_word_prediction"]["expected"]
    predicted = [mask_filling_pipeline(input_text)[0]["token_str"]]
    metrics = {
        "accuracy": accuracy_score(expected, predicted)
    }
    return metrics


# Evaluate Story Prediction
def evaluate_story_prediction():
    input_text = test_data["story_prediction"]["input"]
    generated_story = text_generation_pipeline(input_text, max_length=100)[0]["generated_text"]
    metrics = {"story_length": len(generated_story.split())}
    return metrics


# Evaluate Question Answering
def evaluate_question_answering():
    context = test_data["question_answering"]["input"]["context"]
    question = test_data["question_answering"]["input"]["question"]
    expected_answer = test_data["question_answering"]["expected"]
    generated_answer = qa_pipeline(question=question, context=context)["answer"]
    similarity = SequenceMatcher(None, generated_answer, expected_answer).ratio()
    metrics = {"similarity": similarity, "correct": generated_answer == expected_answer}
    return metrics


# Consolidate metrics and save to JSON
def evaluate_all_tasks():
    results = {
        "text_summarization": evaluate_text_summarization(),
        "sentiment_analysis": evaluate_sentiment(),
        "next_word_prediction": evaluate_next_word_prediction(),
        "story_prediction": evaluate_story_prediction(),
        "question_answering": evaluate_question_answering()
    }
    with open("evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Evaluation metrics saved to evaluation_metrics.json")


# Run the evaluation script
if __name__ == '__main__':
    evaluate_all_tasks()
