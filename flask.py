from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Results page
@app.route('/results', methods=['POST'])
def results():
    def scrape_wikipedia_pages(topic):
        url = f"https://en.wikipedia.org/wiki/{topic}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "bodyContent"})
        line=''
        line += " ".join([p.text for p in content_div.find_all("p")])
        return line
    def preprocess_text(text):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        words = word_tokenize(text)
        english_vocab = set(word.lower() for word in nltk.corpus.words.words())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
        return text
    
    def divide_documents(text):
        paragraphs = text.split('\n\n')
        sentences = []
        for paragraph in paragraphs:
            sentences.extend(sent_tokenize(paragraph))
        return sentences
    def generate_summaries(sentences):
        summaries = []
        for sentence in sentences:
            summary = sentence
            summaries.append(summary)
        return summaries
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    question = request.form['question']
    def answer_questions(summaries, question):
        answers = []
        for summary in summaries:
            ans = qa_pipeline(question=question, context=summary)
            answers.append(ans)
        answer = answers[0]['answer']
        return answer
    
    topic = request.form['topic']
    text = scrape_wikipedia_pages(topic)
    text = preprocess_text(text)
    sentences = divide_documents(text)
    summaries = generate_summaries(sentences)
    answer = answer_questions(summaries, question)
    if answer !=0:
        return render_template('results.html', topic=topic, question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
