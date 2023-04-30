import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Scrape Wikipedia pages
def scrape_wikipedia_pages(topic):
    url = f"https://en.wikipedia.org/wiki/{topic}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    return text

# Preprocess text data
def preprocess_text(text):
    # Remove unwanted characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text into words
    words = word_tokenize(text)
    # Remove stopwords
    english_vocab = set(word.lower() for word in nltk.corpus.words.words())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word in english_vocab]
    # Join the words back into sentences
    text = ' '.join(words)
    return text

# Divide the documents into smaller parts
def divide_documents(text):
    # Divide the text into paragraphs
    paragraphs = text.split('\n\n')
    # Divide each paragraph into sentences
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))
    return sentences

# Generate summaries for each part of the documents using a summarization algorithm
def generate_summaries(sentences):
    # TODO: Implement summarization algorithm
    summaries = []
    for sentence in sentences:
        # TODO: Generate summary for each sentence
        summary = sentence
        summaries.append(summary)
    return summaries

# Use a Question-Answering algorithm to extract the answers from the summarized data for the given questions
model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# set up the pipeline with the specified model
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
def answer_questions(summaries, question):
    # TODO: Implement Question-Answering algorithm
    answers = []
    for summary in summaries:
        # TODO: Extract answer from each summary for the given question
        ans = qa_pipeline(question=question, context=summary)
        answers.append(ans)
    return answers[0]['answer']

# Example usage
topic = "Machine_learning"
text = scrape_wikipedia_pages(topic)
text = preprocess_text(text)
sentences = divide_documents(text)
summaries = generate_summaries(sentences)
question = "What are the approaches?"
answers = answer_questions(summaries, question)
print(question)
print("Answer: " ,answers)
