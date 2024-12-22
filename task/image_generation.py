from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt):
    try:
        # Load the Stable Diffusion pipeline (make sure to use the correct model)
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-original", torch_dtype=torch.float16)
        pipe.to("cuda")  # Use GPU if available, or remove if using CPU

        # Generate image based on the text prompt
        image = pipe(prompt).images[0]

        return image

    except ValueError as ve:
        raise ValueError("Invalid input prompt for image generation") from ve

    except Exception as e:
        raise RuntimeError(f"An error occurred during image generation: {e}") from e
