from diffusers import StableDiffusionPipeline
import torch
import json
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image


# Function to generate images from prompts
def generate_image(prompt):
    try:
        # Load the Stable Diffusion pipeline (make sure to use the correct model)
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-original",
                                                       torch_dtype=torch.float16)
        pipe.to("cuda")  # Use GPU if available, or remove if using CPU

        # Generate image based on the text prompt
        image = pipe(prompt).images[0]

        return image

    except ValueError as ve:
        raise ValueError("Invalid input prompt for image generation") from ve

    except Exception as e:
        raise RuntimeError(f"An error occurred during image generation: {e}") from e


# Function to calculate image similarity (using SSIM as an example)
def calculate_similarity(expected_image_path, generated_image):
    try:
        # Open expected image
        expected_image = Image.open(expected_image_path).convert('RGB')
        expected_image = np.array(expected_image)

        # Convert generated image to numpy array
        generated_image = np.array(generated_image)

        # Compute SSIM (you can replace this with another metric if desired)
        similarity_index, _ = ssim(expected_image, generated_image, full=True, multichannel=True)

        return similarity_index

    except Exception as e:
        raise RuntimeError(f"An error occurred during similarity computation: {e}") from e


# Function to evaluate image generation performance
def evaluate_image_generation(test_data, results_file="task_results.json"):
    task_name = "image_generation"
    similarity_scores = []

    # Iterate over the test dataset
    for sample in test_data:
        prompt = sample["input"]
        expected_image_path = sample["expected"]

        # Generate image from prompt
        generated_image = generate_image(prompt)

        # Calculate similarity (using SSIM as an example)
        similarity = calculate_similarity(expected_image_path, generated_image)
        similarity_scores.append(similarity)

    # Calculate average similarity score
    avg_similarity = sum(similarity_scores) / len(similarity_scores)

    # Save results to JSON file
    save_results_to_json(task_name, {"average_similarity": avg_similarity}, results_file)

    # Print the final evaluation metrics
    print(f"\nFinal Metrics for image generation task: '{task_name}'")
    print(f"Average Similarity: {avg_similarity:.2f}")

    return avg_similarity


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


# Test data for image generation task
test_data = [
    {
        "input": "A beautiful landscape with mountains and a river",
        "expected": "path/to/expected_image1.png"  # Path to expected image
    },
    {
        "input": "A futuristic city with flying cars",
        "expected": "path/to/expected_image2.png"  # Path to expected image
    }
]

# Evaluate the image generation task
evaluate_image_generation(test_data)
