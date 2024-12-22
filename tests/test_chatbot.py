from nltk.translate.bleu_score import sentence_bleu
import json
import os

# Function to simulate chatbot response (replace this with actual chatbot logic)
def chatbot_response(user_input):
    # Dummy chatbot response logic for testing (replace with actual chatbot logic)
    return "I am your chatbot, how can I assist you?"

# Function to save results to a JSON file
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

# Function to evaluate chatbot performance using BLEU score
def evaluate_chatbot(test_data, results_file="task_results.json"):
    task_name = "chatbot"
    bleu_scores = []

    for sample in test_data:
        user_input = sample["input"]
        expected_response = sample["expected"]
        chatbot_response_text = chatbot_response(user_input)  # Get chatbot's response

        # Calculate BLEU score between expected response and chatbot response
        reference = [expected_response.split()]
        candidate = chatbot_response_text.split()
        score = sentence_bleu(reference, candidate)
        bleu_scores.append(score)

        # Print individual task evaluation for debugging
        print(f"User Input: {user_input}")
        print(f"Expected Response: {expected_response}")
        print(f"Chatbot Response: {chatbot_response_text}")
        print(f"BLEU Score: {score:.4f}")

    # Calculate the average BLEU score
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # Save the results to the specified file
    save_results_to_json(task_name, {"average_bleu": avg_bleu}, results_file)

    # Print final evaluation metrics
    print(f"\nFinal Metrics for chatbot task: '{task_name}'")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    return avg_bleu

# Test data for the chatbot evaluation task
test_data = [
    {"input": "Hello, how are you?", "expected": "I'm doing well, thank you!"},
    {"input": "What is your name?", "expected": "I am a chatbot, and I don't have a name."},
    {"input": "Can you help me with coding?", "expected": "Sure! What kind of coding help do you need?"}
]

# Path to the existing results file
results_file = "D:\\pythonProject\\pythonProject\\nlp_project_work\\tests\\task_results.json"

# Evaluate the chatbot task and append the results
evaluate_chatbot(test_data, results_file)
