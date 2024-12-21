# Pratapgh-multifunctional-nlp-image-generation-tool-
The Multifunctional NLP and Image Generation Tool integrates Hugging Face’s pretrained models to offer various NLP tasks and image generation. With an intuitive Streamlit interface, it supports text summarization, next word prediction, story generation, chatbot interaction, sentiment analysis, and more


```markdown
# Multifunctional NLP and Image Generation Tool using Hugging Face Models

A multifunctional tool using Hugging Face models for NLP tasks like text summarization, sentiment analysis, and chatbot functionality, along with image generation. This project integrates various AI models into a single application with a user-friendly interface built with Streamlit.

## Project Overview
This project implements a **Multifunctional NLP and Image Generation Tool** that leverages pretrained models from **Hugging Face** for performing various Natural Language Processing (NLP) tasks and generating images. The tool provides a user-friendly interface, enabling users to interact seamlessly with various features:

- Text Summarization
- Next Word Prediction
- Story Prediction
- Chatbot Interaction
- Sentiment Analysis
- Question Answering
- Image Generation

The project is developed with a modular structure for maintainability and scalability, utilizing **Streamlit** for the frontend and robust backend integrations for multiple AI tasks.

## Project Features
- **Streamlit-Based UI**: An intuitive interface allowing users to select and execute tasks.
- **Pretrained Models**: Utilizes state-of-the-art Hugging Face models for various NLP and image generation tasks.
- **Error Handling**: Comprehensive exception handling for smoother user experience.
- **Task Variety**: Covers essential NLP functionalities alongside image generation.
- **Logging**: Integrated logging for performance tracking and debugging.

## Key Skills Gained
- Efficient use of **Hugging Face Transformers** for diverse tasks.
- Development of interactive applications using **Streamlit**.
- Modular design for scalable application development.
- Handling real-time user interactions and inputs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

## Installation
Follow these steps to set up and run the project on **Windows 10** using **PyCharm**:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/multifunctional-nlp-image-generation-tool.git
   ```

2. **Navigate to the Project Folder**:
   ```bash
   cd multifunctional-nlp-image-generation-tool
   ```

3. **Set Up a Virtual Environment**:
   - In PyCharm:
     - Go to `File > Settings > Project: <Your Project> > Python Interpreter`.
     - Click on the gear icon and select `Add...`.
     - Create a new virtual environment.
   - Alternatively, use the terminal:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**:
   - In PyCharm:
     - Open `app.py` and click the run button.
   - Using the terminal:
     ```bash
     streamlit run app.py
     ```

6. **Access the Application**:
   Open `http://localhost:8501` in your browser.

## Usage
### Available Tasks
- **Text Summarization**: Provide text and generate a concise summary.
- **Next Word Prediction**: Predict the next word in a sentence containing the `[MASK]` token.
- **Story Prediction**: Extend a story based on the given prompt.
- **Chatbot Interaction**: Engage in a conversation with the chatbot.
- **Sentiment Analysis**: Analyze the sentiment (positive/negative/neutral) of a text.
- **Question Answering**: Answer questions based on a provided context.
- **Image Generation**: Generate an image from a textual description.

## Project Structure
```
multifunctional-nlp-image-generation-tool/
├── app.py                   # Main Streamlit app with task integration
├── requirements.txt         # List of required dependencies
├── tasks/                   # Folder containing task-specific logic
│   ├── __init__.py
│   ├── text_summarization.py
│   ├── next_word_prediction.py
│   ├── story_prediction.py
│   ├── chatbot.py
│   ├── sentiment_analysis.py
│   ├── question_answering.py
│   └── image_generation.py
├── models/                  # Directory for model initialization
│   ├── __init__.py
│   ├── huggingface_loader.py
│   └── tokenizer.py
├── config/                  # Directory for configuration files
│   ├── __init__.py
│   ├── settings.py
│   └── constants.py
├── logs/                    # Logs for debugging and tracking
│   └── app.log
├── tests/                   # Test folder with task-specific unit tests
│   ├── __init__.py
│   ├── test_chatbot.py
│   ├── test_image_generation.py
│   ├── test_next_word_prediction.py
│   ├── test_question_answering.py
│   ├── test_sentiment_analysis.py
│   ├── test_story_prediction.py
│   ├── test_text_summarization.py
│   └── task_results.json      # Stores task evaluation results
└── README.md                # Project documentation
```

## Evaluation Metrics
### Task Performance

| Task                  | Accuracy | Precision | Recall | F1-Score | ROUGE-1 Recall | ROUGE-1 Precision | ROUGE-1 F1 | ROUGE-2 Recall | ROUGE-2 Precision | ROUGE-2 F1 | ROUGE-L Recall | ROUGE-L Precision | ROUGE-L F1 | Top-K Accuracy | BLEU Score | Exact Match |
|-----------------------|----------|-----------|--------|----------|----------------|-------------------|------------|----------------|-------------------|------------|----------------|-------------------|------------|-----------------|------------|-------------|
| Next Word Prediction  | 33.33%   | 33.33%    | 33.33% | 33.33%   | -              | -                 | -          | 66.67%         | -                 | -          | -              | -                 | -          | -               | -          | -           |
| Text Summarization    | -        | -         | -      | -        | 54.23%         | 24.29%            | 33.30%     | 16.04%         | 6.07%             | 8.73%      | 48.16%         | 21.46%            | 29.46%     | -               | -          | -           |
| Story Prediction      | 0.00%    | 16.67%    | 1.72%  | -        | -              | -                 | -          | -              | -                 | -          | -              | -                 | -          | -               | -          | -           |
| Chatbot               | -        | -         | -      | -        | -              | -                 | -          | -              | -                 | -          | -              | -                 | -          | -               | 2.16e-155  | -           |
| Sentiment Analysis    | 100%   | 100%    | 100% | 100%   | -              | -                 | -          |               | -                 | -          | -              | -                 | -          | -               | -          | 100%         |
| Question Answering    | -        | -         | -      | -        | -              | -                 | -          | -              | -                 | -          | -              | -                 | -          | -               | -          | 100%        |

Performance is validated through manual testing and feedback collection.

## Testing
### Manual Testing
Tested each task with valid and invalid inputs to verify:
- Accuracy of results.
- Robustness of error handling.

### Automated Testing
Incorporated unit tests for key functionalities in the `tests/` folder.

### Sample Test Cases
- **Text Summarization**: Provide a passage and validate the summary.
- **Sentiment Analysis**: Test various sentiments for correctness.
- **Question Answering**: Ask context-based questions to verify answers.
- **Image Generation**: Generate images based on descriptive prompts.

## Future Improvements
1. **Cloud Deployment**: Host the app on platforms like AWS or Azure.
2. **Model Fine-Tuning**: Adapt models to domain-specific datasets.
3. **Task Expansion**: Add new NLP tasks like translation and paraphrasing.
4. **Optimization**: Reduce response times for computationally intensive tasks.
5. **Enhanced UI**: Add real-time previews and performance metrics displays.

## Contributing
Contributions are welcome! If you wish to contribute:
1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Submit a pull request with a clear description of your changes.
```

