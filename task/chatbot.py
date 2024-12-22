from transformers import pipeline

def chatbot_response(prompt):
    try:
        # Load the text-generation pipeline with GPT-2
        chatbot = pipeline("text-generation", model="gpt2")

        # Generate a response with controlled parameters
        response = chatbot(
            prompt,
            max_length=50,          # Reduced max length for more concise responses
            num_return_sequences=1, # Only return one response for simplicity
            temperature=0.5,        # Lower temperature for more deterministic responses
            top_k=30,               # Further reduced to make output more focused
            top_p=0.85,             # Lower nucleus sampling for better control
            repetition_penalty=1.2, # Penalizes repeating phrases for more diverse responses
            no_repeat_ngram_size=3, # Prevents repeating n-grams, ensuring more diverse text
            pad_token_id=50256,     # Token for padding to ensure proper response formatting
            truncation=True         # Explicit truncation to prevent warnings
        )

        # Get the generated text
        generated_text = response[0]['generated_text']

        # Post-processing: Cut the response to focus on the direct reply
        response_cut = generated_text.split('.')[0]  # Take the first sentence
        return response_cut.strip()

    except ValueError as ve:
        raise ValueError("Invalid input prompt for chatbot") from ve

    except Exception as e:
        raise RuntimeError("An error occurred during chatbot processing") from e

# Test the chatbot
test_inputs = [
    "Hello, good morning!",
    "What is your name?",
    "Can you help me with coding?"
]

for prompt in test_inputs:
    print(f"Input: {prompt}")
    response = chatbot_response(prompt)
    print(f"Chatbot Response: {response}\n")
