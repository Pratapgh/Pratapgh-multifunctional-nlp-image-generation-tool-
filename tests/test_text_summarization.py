from transformers import pipeline
from rouge import Rouge
import json
import os

# Function to summarize text using Hugging Face's summarization pipeline
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

# Function to evaluate summarization performance using ROUGE scores
def evaluate_text_summarization(test_data, results_file):
    task_name = "text_summarization"
    rouge = Rouge()
    y_true = []  # Ground truth summaries
    y_pred = []  # Model-generated summaries

    # Iterate over the test dataset
    for sample in test_data:
        text = sample["input"]
        expected_summary = sample["expected"]
        generated_summary = summarize_text(text)  # Generate summary using the summarize_text function
        y_true.append(expected_summary)
        y_pred.append(generated_summary)

    # Calculate ROUGE scores
    rouge_scores = rouge.get_scores(y_pred, y_true, avg=True)

    # Save results to the specified file
    save_results_to_json(task_name, rouge_scores, results_file)

    # Print final evaluation metrics
    print(f"\nFinal Metrics for text summarization task: '{task_name}'")
    print(f"ROUGE-1: {rouge_scores['rouge-1']}")
    print(f"ROUGE-2: {rouge_scores['rouge-2']}")
    print(f"ROUGE-L: {rouge_scores['rouge-l']}")
    return rouge_scores

# Function to append or update evaluation results in the task_results.json file
def save_results_to_json(task_name, results, results_file):
    # Ensure the results file exists or create it
    if not os.path.exists(results_file):
        with open(results_file, "w") as file:
            json.dump({}, file)

    # Read the existing results from the file
    try:
        with open(results_file, "r") as file:
            all_results = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    # Add or update the results for the current task
    all_results[task_name] = results

    # Write the updated results back to the file
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

# Test data for summarization task
test_data = [
    {
        "input": """
        Hugging Face is a company that provides machine learning tools for NLP tasks. 
        It has become one of the most well-known organizations in the NLP space due to its popular open-source libraries and models, 
        such as BERT, GPT-2, and many others. The company has been building cutting-edge tools for transforming and 
        fine-tuning models to solve real-world problems in NLP.
        """,
        "expected": "Hugging Face is a leading company in NLP, known for its open-source models like BERT and GPT-2."
    },
    {
        "input": """
        OpenAI is an artificial intelligence research organization that aims to ensure that artificial general intelligence (AGI) benefits all of humanity. 
        The organization has released models like GPT-3, which have been influential in the AI community.
        """,
        "expected": "OpenAI is focused on ensuring that AGI benefits all of humanity, with influential models like GPT-3."
    }
]

# Path to the existing results file
results_file = "D:\\pythonProject\\pythonProject\\nlp_project_work\\tests\\task_results.json"

# Evaluate the summarization task and append the results
evaluate_text_summarization(test_data, results_file)
