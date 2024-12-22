import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import pipeline

# Test dataset: input sentences and expected predictions
test_data = [
    {"input": "I love your [MASK].", "expected": "smile"},
    {"input": "What is your [MASK]?", "expected": "name"},
    {"input": "She is the [MASK] of the class.", "expected": "best"},
]

# Function to evaluate model performance and save results
def evaluate_and_save_results(test_data, top_k=5, results_file="task_results.json"):
    try:
        # Dynamically get task name from script name
        task_name = os.path.splitext(os.path.basename(__file__))[0]  # Get filename without extension

        # Load the fill-mask pipeline
        predictor = pipeline("fill-mask", model="bert-base-uncased")

        print(f"Starting evaluation of model performance for '{task_name}' task...\n")

        y_true = []  # Ground truth labels
        y_pred = []  # Model's top-1 predictions
        top_k_correct = 0  # Count of correct predictions in top-k
        detailed_results = []  # Store detailed results for each sample

        # Iterate over test data
        for index, sample in enumerate(test_data, start=1):
            input_text = sample["input"]
            expected = sample["expected"]

            # Predict using the pipeline
            predictions = predictor(input_text)
            top_predictions = [pred["token_str"] for pred in predictions]

            # Append ground truth and top-1 prediction
            y_true.append(expected)
            y_pred.append(top_predictions[0])  # Top-1 prediction

            # Check if the expected word is in top-k predictions
            if expected in top_predictions[:top_k]:
                top_k_correct += 1

            # Append detailed results
            detailed_results.append({
                "sample_id": index,
                "input": input_text,
                "expected": expected,
                "top_1_prediction": top_predictions[0],
                "top_k_predictions": top_predictions[:top_k],
            })

            # Print predictions for each input
            print(f"Sample {index}:")
            print(f"Input: {input_text}")
            print(f"Expected: '{expected}'")
            print(f"Top-1 Prediction: '{top_predictions[0]}'")
            print(f"Top-{top_k} Predictions: {top_predictions[:top_k]}\n")

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        top_k_accuracy = top_k_correct / len(test_data)

        # Prepare results to save
        final_results = {
            "task_name": task_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "top_k_accuracy": top_k_accuracy,
            "detailed_results": detailed_results,
        }

        # Save results to JSON file
        try:
            with open(results_file, "r") as file:
                all_results = json.load(file)  # Load existing results
        except FileNotFoundError:
            all_results = {}  # Create a new file if it doesn't exist

        all_results[task_name] = final_results

        with open(results_file, "w") as file:
            json.dump(all_results, file, indent=4)

        print(f"Results saved successfully to '{results_file}'.")

        return final_results

    except Exception as e:
        raise RuntimeError(f"An error occurred during '{task_name}' evaluation") from e


# Run evaluation and save results
if __name__ == "__main__":
    results = evaluate_and_save_results(test_data)

    # Print the final metrics dynamically for the task
    print(f"\nFinal Metrics for '{results['task_name']}' task:")
    print(f"Accuracy: {results['accuracy']:.2f}")
    print(f"Precision: {results['precision']:.2f}")
    print(f"Recall: {results['recall']:.2f}")
    print(f"F1-score: {results['f1_score']:.2f}")
    print(f"Top-5 Accuracy: {results['top_k_accuracy']:.2f}")
