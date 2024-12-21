import streamlit as st
from tasks.text_summarization import summarize_text
from tasks.next_word_prediction import predict_next_word
from tasks.story_prediction import predict_story
from tasks.chatbot import chatbot_response
from tasks.sentiment_analysis import analyze_sentiment
from tasks.question_answering import answer_question
from tasks.image_generation import generate_image

# Set up the Streamlit page
st.set_page_config(page_title="Multifunctional NLP and Image Generation Tool", layout="wide")

# Sidebar for task selection
st.sidebar.title("Task Selector")
task = st.sidebar.selectbox(
    "Select a Task",
    [
        "Text Summarization",
        "Next Word Prediction",
        "Story Prediction",
        "Chatbot",
        "Sentiment Analysis",
        "Question Answering",
        "Image Generation"
    ]
)

# Task-specific input forms and processing
if task == "Text Summarization":
    st.header("Text Summarization")
    text = st.text_area("Enter text to summarize", "")
    if st.button("Summarize"):
        if text:
            try:
                summary = summarize_text(text)
                st.success("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter text to summarize.")

elif task == "Next Word Prediction":
    st.header("Next Word Prediction")
    prompt = st.text_input("Enter a prompt", "")
    if st.button("Predict Next Word"):
        if prompt:
            
            try:
                prediction = predict_next_word(prompt)
                st.success("Predicted Text:")
                st.write(prediction)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a prompt.")

elif task == "Story Prediction":
    st.header("Story Prediction")
    prompt = st.text_input("Enter a story prompt", "")
    if st.button("Generate Story"):
        if prompt:
            try:
                story = predict_story(prompt)
                st.success("Generated Story:")
                st.write(story)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a story prompt.")

elif task == "Chatbot":
    st.header("Chatbot")
    user_input = st.text_input("Enter your message", "")
    if st.button("Get Response"):
        if user_input:
            try:
                response = chatbot_response(user_input)
                st.success("Chatbot Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a message.")

elif task == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text = st.text_area("Enter text to analyze sentiment", "")
    if st.button("Analyze Sentiment"):
        if text:
            try:
                sentiment = analyze_sentiment(text)
                st.success("Sentiment Result:")
                st.write(sentiment)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter text for sentiment analysis.")

elif task == "Question Answering":
    st.header("Question Answering")
    question = st.text_input("Enter your question", "")
    context = st.text_area("Provide context", "")
    if st.button("Get Answer"):
        if question and context:
            try:
                answer = answer_question(question, context)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter both a question and context.")

elif task == "Image Generation":
    st.header("Image Generation")
    prompt = st.text_input("Enter an image description", "")
    if st.button("Generate Image"):
        if prompt:
            try:
                # Assuming the output of generate_image is an image URL or file path
                image = generate_image(prompt)
                st.success("Generated Image:")
                st.image(image)  # Adjust as per your pipeline output
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter an image description.")

