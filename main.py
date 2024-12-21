import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evaluate import load
from transformers import pipeline

# For Text Summarization, using ROUGE metric
rouge = load("rouge")

# Define task-specific functions directly in the script

# Text Summarizer function
summarizer = pipeline("summarization")

# Sentiment Analyzer function
sentiment_analyzer = pipeline("sentiment-analysis")

# Question Answering model function
qa_model = pipeline("question-answering")

# Next Word Predictor function
next_word_predictor = pipeline("text-generation")

# Story Generator function
story_predictor = pipeline("text-generation")

# Chatbot function
chatbot_model = pipeline("conversational")

# Image Generator function
image_generator = pipeline("image-generation")

# Function to measure response time
def measure_response_time(task_function, *args):
    start_time = time.time()
    result = task_function(*args)
    end_time = time.time()
    return result, end_time - start_time

# Text Summarization Test
def test_text_summarization():
    texts = ["The quick brown fox jumps over the lazy dog."]
    references = ["A quick fox jumps over a lazy dog."]

    # Generate summaries
    predictions = [summarizer(text)[0]['summary_text'] for text in texts]

    # Compute ROUGE score
    results = rouge.compute(predictions=predictions, references=references)

    # Measure response time
    _, response_time = measure_response_time(summarizer, texts[0])
    return {"rouge": results, "response_time": response_time}

# Sentiment Analysis Test
def test_sentiment_analysis():
    texts = ["I love this!", "I hate this."]
    true_labels = ["POSITIVE", "NEGATIVE"]

    # Predict sentiments
    predictions = [sentiment_analyzer(text)[0]['label'] for text in texts]

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label="POSITIVE")
    recall = recall_score(true_labels, predictions, pos_label="POSITIVE")
    f1 = f1_score(true_labels, predictions, average='weighted')

    # Measure response time
    _, response_time = measure_response_time(sentiment_analyzer, texts[0])
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
            "response_time": response_time}

# Question Answering Test
def test_question_answering():
    context = "The Eiffel Tower is located in Paris."
    questions = ["Where is the Eiffel Tower located?"]
    true_answers = ["Paris"]

    # Predict answers
    predictions = [qa_model(context=context, question=question)[0]['answer'] for question in questions]

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(true_answers, predictions)
    precision = precision_score(true_answers, predictions, average='micro')
    recall = recall_score(true_answers, predictions, average='micro')
    f1 = f1_score(true_answers, predictions, average='micro')

    # Measure response time
    _, response_time = measure_response_time(qa_model, context, questions[0])
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
            "response_time": response_time}

# Next Word Prediction Test
def test_next_word_prediction():
    prompt = "The quick brown fox jumps"
    true_next_word = "over"

    # Predict next word
    prediction = next_word_predictor(prompt)[0]['generated_text']
    predicted_word = prediction.split()[-1]

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = 1 if predicted_word == true_next_word else 0
    precision = accuracy
    recall = accuracy
    f1 = f1_score([true_next_word], [predicted_word], average='binary')

    # Measure response time
    _, response_time = measure_response_time(next_word_predictor, prompt)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
            "response_time": response_time}

# Story Prediction Test
def test_story_prediction():
    prompt = "Once upon a time, in a faraway land, there was a"
    true_story_end = "king."

    # Generate story continuation
    prediction = story_predictor(prompt)[0]['generated_text']
    predicted_story_end = prediction.split()[-1]

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = 1 if predicted_story_end == true_story_end else 0
    precision = accuracy
    recall = accuracy
    f1 = f1_score([true_story_end], [predicted_story_end], average='binary')

    # Measure response time
    _, response_time = measure_response_time(story_predictor, prompt)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
            "response_time": response_time}

# Chatbot Test
def test_chatbot():
    prompt = "Hello, how are you?"
    true_response = "I'm good, thank you!"

    # Generate chatbot response
    response = chatbot_model(prompt)[0]['generated_text']

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = 1 if response == true_response else 0
    precision = accuracy
    recall = accuracy
    f1 = f1_score([true_response], [response], average='binary')

    # Measure response time
    _, response_time = measure_response_time(chatbot_model, prompt)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
            "response_time": response_time}

# Image Generation Test
def test_image_generation():
    prompt = "A futuristic cityscape at sunset."

    # Measure response time for image generation
    _, response_time = measure_response_time(image_generator, prompt)
    return {"response_time": response_time}

# Main function to run all tests
def run_all_tests():
    results = {}

    print("Running Text Summarization Test...")
    results['Text Summarization'] = test_text_summarization()

    print("Running Sentiment Analysis Test...")
    results['Sentiment Analysis'] = test_sentiment_analysis()

    print("Running Question Answering Test...")
    results['Question Answering'] = test_question_answering()

    print("Running Next Word Prediction Test...")
    results['Next Word Prediction'] = test_next_word_prediction()

    print("Running Story Prediction Test...")
    results['Story Prediction'] = test_story_prediction()

    print("Running Chatbot Test...")
    results['Chatbot'] = test_chatbot()

    print("Running Image Generation Test...")
    results['Image Generation'] = test_image_generation()

    for task, result in results.items():
        print(f"\n{task} Results:")
        for metric, value in result.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    print(f"  {sub_metric}: {sub_value}")
            else:
                print(f"  {metric}: {value:.2f}" if isinstance(value, (float, int)) else f"  {metric}: {value}")

    return results

def save_results_to_json(results, filename="test_results.json"):
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

# In your main function, after running the tests:
if __name__ == "__main__":
    final_results = run_all_tests()
    # Save results to a JSON file
    save_results_to_json(final_results)
