from tasks.text_summarization import summarize_text
from tasks.image_generation import generate_image
from tasks.chatbot import chatbot_response
from tasks.sentiment_analysis import analyze_sentiment
from tasks.next_word_prediction import predict_next_word
from tasks.question_answering import answer_question
from tasks.story_prediction import predict_story


def test_text_summarization():
    print("Testing Text Summarization...")
    try:
        # Valid input
        print(summarize_text("Artificial intelligence is transforming the world."))

        # Invalid input (empty text)
        print(summarize_text(""))
    except Exception as e:
        print(e)


def test_image_generation():
    print("\nTesting Image Generation...")
    try:
        # Valid input
        print(generate_image("A futuristic city in the clouds"))

        # Invalid input (empty prompt)
        print(generate_image(""))
    except Exception as e:
        print(e)


def test_chatbot():
    print("\nTesting Chatbot...")
    try:
        # Valid input
        print(chatbot_response("What is the weather today?"))

        # Invalid input (empty prompt)
        print(chatbot_response(""))
    except Exception as e:
        print(e)


def test_sentiment_analysis():
    print("\nTesting Sentiment Analysis...")
    try:
        # Valid input
        print(analyze_sentiment("I love programming with Python."))

        # Invalid input (non-text input)
        print(analyze_sentiment(12345))
    except Exception as e:
        print(e)


def test_next_word_prediction():
    print("\nTesting Next Word Prediction...")
    try:
        # Valid input
        print(predict_next_word("The cat sat on the [MASK]."))

        # Invalid input (no [MASK])
        print(predict_next_word("The cat sat on the mat."))
    except Exception as e:
        print(e)


def test_question_answering():
    print("\nTesting Question Answering...")
    try:
        # Valid input
        context = "The Eiffel Tower is in Paris, France."
        question = "Where is the Eiffel Tower located?"
        print(answer_question(context, question))

        # Invalid input (empty context or question)
        print(answer_question("", "Where is the Eiffel Tower located?"))
    except Exception as e:
        print(e)


def test_story_prediction():
    print("\nTesting Story Prediction...")
    try:
        # Valid input
        print(predict_story("Once upon a time in a distant galaxy,"))

        # Invalid input (empty prompt)
        print(predict_story(""))
    except Exception as e:
        print(e)


# Run all tests
if __name__ == "__main__":
    test_text_summarization()
    test_image_generation()
    test_chatbot()
    test_sentiment_analysis()
    test_next_word_prediction()
    test_question_answering()
    test_story_prediction()
