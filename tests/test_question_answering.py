from transformers import pipeline
import json
from tasks.question_answering import answer_question  # Import your pre-existing function


# Function to evaluate question answering performance
def evaluate_question_answering(test_data, results_file="task_results.json"):
    task_name = "question_answering"
    em_scores = []  # Exact Match scores
    f1_scores = []  # F1 scores

    # Iterate over the test dataset
    for sample in test_data:
        question = sample["input"]["question"]
        context = sample["input"]["context"]
        expected_answer = sample["expected"]

        # Call your pre-existing answer_question function
        predicted_answer = answer_question(question, context)

        # Exact Match
        em = int(predicted_answer.strip().lower() == expected_answer.strip().lower())
        em_scores.append(em)

        # F1-score
        true_set = set(expected_answer.split())
        pred_set = set(predicted_answer.split())
        common = true_set.intersection(pred_set)
        f1 = 2 * len(common) / (len(true_set) + len(pred_set))
        f1_scores.append(f1)

    # Calculate average EM and F1 scores
    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    # Save results to JSON file
    save_results_to_json(task_name, {"exact_match": avg_em, "f1_score": avg_f1}, results_file)

    # Print the final evaluation metrics
    print(f"\nFinal Metrics for question answering task: '{task_name}'")
    print(f"Exact Match: {avg_em:.2f}")
    print(f"F1-score: {avg_f1:.2f}")

    return avg_em, avg_f1


# Function to save evaluation results to a JSON file
def save_results_to_json(task_name, results, results_file):
    # Read the existing results if the file exists
    try:
        with open(results_file, "r") as file:
            all_results = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    # Add new task results
    all_results[task_name] = results

    # Save updated results back to the file
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)


# Test data for question answering task
test_data = [
    {
        "input": {
            "question": "Who developed the theory of relativity?",
            "context": "Albert Einstein developed the theory of relativity, which revolutionized modern physics."
        },
        "expected": "Albert Einstein"
    },
    {
        "input": {
            "question": "What is the capital of France?",
            "context": "The capital of France is Paris, which is also known for its art, culture, and landmarks."
        },
        "expected": "Paris"
    }
]

# Evaluate the question answering task
evaluate_question_answering(test_data)
