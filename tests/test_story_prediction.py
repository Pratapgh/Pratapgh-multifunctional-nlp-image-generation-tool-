from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import os

# Story prediction function (to be implemented)
def predict_story(prompt):
    # Replace this with your actual story prediction logic
    return "This is the predicted story based on the prompt."

# Function to evaluate story prediction performance using BLEU scores
def evaluate_story_prediction(test_data, results_file="task_results.json"):
    task_name = "story_prediction"
    bleu_scores = []

    for sample in test_data:
        prompt = sample["input"]
        expected_story = sample["expected"]
        generated_story = predict_story(prompt)

        # Debugging: Print reference and candidate
        print("Reference (expected):", expected_story)
        print("Candidate (generated):", generated_story)

        # Calculate BLEU score with smoothing
        reference = [expected_story.split()]
        candidate = generated_story.split()
        smoothing_function = SmoothingFunction().method1
        score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
        bleu_scores.append(score)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # Save results
    save_results_to_json(task_name, {"average_bleu": avg_bleu, "bleu_scores": bleu_scores}, results_file)
    print(f"\nFinal Metrics for story prediction task: '{task_name}'")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    return avg_bleu


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

# Test data for story prediction task
test_data = [
    {
        "input": "Once upon a time, there was a brave knight who",
        "expected":  "Once upon a time, there was a brave knight who fought dragons to save the kingdom. "
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



