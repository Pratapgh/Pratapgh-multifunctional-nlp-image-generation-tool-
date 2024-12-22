import os
import json
from sklearn.metrics import precision_score, recall_score


# Story prediction function (to be implemented)
def predict_story(prompt):
    # Replace this with your actual story prediction logic
    return "This is the predicted story based on the prompt."


def evaluate_story_prediction(test_data, results_file="task_results.json"):
    task_name = "story_prediction"
    total_correct = 0
    total_generated_tokens = 0
    total_reference_tokens = 0
    correct_tokens = 0

    for sample in test_data:
        prompt = sample["input"]
        expected_story = sample["expected"]
        generated_story = predict_story(prompt)

        # Tokenize the expected and generated stories
        expected_tokens = expected_story.split()
        generated_tokens = generated_story.split()

        # Calculate exact match (accuracy)
        if generated_story.strip() == expected_story.strip():
            total_correct += 1

        # Count tokens for precision and recall
        total_generated_tokens += len(generated_tokens)
        total_reference_tokens += len(expected_tokens)
        correct_tokens += len(set(generated_tokens) & set(expected_tokens))  # Intersection of words

    # Calculate Accuracy
    accuracy = total_correct / len(test_data) if len(test_data) > 0 else 0

    # Calculate Precision and Recall (token-level)
    precision = correct_tokens / total_generated_tokens if total_generated_tokens > 0 else 0
    recall = correct_tokens / total_reference_tokens if total_reference_tokens > 0 else 0

    # Save results
    save_results_to_json(task_name, {"accuracy": accuracy, "precision": precision, "recall": recall}, results_file)
    print(f"\nFinal Metrics for story prediction task: '{task_name}'")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return accuracy, precision, recall


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


# Test data for story prediction task
test_data = [
    {
        "input": "Once upon a time, there was a brave knight who",
        "expected": "Once upon a time, there was a brave knight who fought dragons to save the kingdom. "
                    "But before he had a chance to learn the true meaning of his words, the great war had been waged against dragons. "
                    "But it had happened, he realized. It wasn't the dragons who had come to the Dragon Kingdom but the people who had fought against them. "
                    "That group of people wasn't the greatest warriors of dragons but there was still time. It was the King of Dragons. "
                    "A hero once said that there were many heroes but as a king, the true strength of a king is far more than one single warrior. "
                    "He was the Prince of the Sun. 'Lord!' This time, the Emperor's voice was even more deep. "
                    "A true king had no choice but to fight. His victory would have meant that his opponent would have lost his battle against darkness. "
                    "At the same time, his victory would not mean"
    },
    {
        "input": "In a distant galaxy, a young explorer discovered",
        "expected": "In a distant galaxy, a young explorer discovered a mysterious planet with intelligent life."
    }
]

# Path to the existing results file
results_file = "D:\\pythonProject\\pythonProject\\nlp_project_work\\tests\\task_results.json"

# Evaluate the story prediction task and append the results
evaluate_story_prediction(test_data, results_file)
