import os
import json
from sklearn.metrics import precision_recall_fscore_support
from tasks.sentiment_analysis import analyze_sentiment

def test_sentiment_analysis():
    test_texts = ["I love this!", "I hate this."]
    expected_labels = ["POSITIVE", "NEGATIVE"]

    # Get predictions from the analyze_sentiment function
    predictions = [analyze_sentiment(text)['label'] for text in test_texts]

    # Calculate accuracy
    accuracy = sum(p == e for p, e in zip(predictions, expected_labels)) / len(expected_labels)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected_labels, predictions, average="binary", pos_label="POSITIVE"
    )

    # Prepare the results with task name and evaluation metrics
    results = {
        "task_name": "sentiment_analysis",  # Task name
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Path to store results
    results_file = r"D:\pythonProject\pythonProject\nlp_project_work\tests\task_results.json"

    # Load existing results from task_results.json if it exists
    try:
        with open(results_file, "r") as file:
            all_results = json.load(file)  # Load existing results
    except FileNotFoundError:
        all_results = {}  # If file doesn't exist, create a new dictionary

    # Add the new test results to the existing ones
    all_results[results["task_name"]] = results

    # Save the updated results back to the JSON file
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    # Print the results
    print("Sentiment Analysis Results:")
    print(f"Task: {results['task_name']}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    return results


if __name__ == "__main__":
    test_sentiment_analysis()
