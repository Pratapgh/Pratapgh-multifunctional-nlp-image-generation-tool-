from transformers import pipeline


def predict_story(prompt):
    try:
        # Load the story generation pipeline
        story_generator = pipeline("text-generation", model="gpt2")

        # Generate a story
        story = story_generator(prompt, max_length=200, num_return_sequences=1)
        return story[0]['generated_text']

    except ValueError as ve:
        raise ValueError("Invalid input prompt for story prediction") from ve

    except Exception as e:
        raise RuntimeError("An error occurred during story prediction") from e
