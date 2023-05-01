Wikipedia Summarization and Question Answering
This is a Python script that allows you to extract summaries and answers from Wikipedia pages. The script is divided into several functions, each responsible for a specific task.



NOTE!!!! : The code for generating multiple-document summary and Question-Answering model is also ready, please try it once in the python IDE once before trying in the flask server.



Installation

The following packages are required to run the script:

requests
bs4
nltk
transformers

To install these packages, you can use the following command:


pip install requests bs4 nltk transformers
import nltk
nltk.download('words')
nltk.download('punkt')
Usage
To use the script, follow these steps:

Run the script using python test.py command.
Enter the number of topics you want to summarize and ask questions about.
For each topic, enter the name of the topic as it appears in the Wikipedia URL (e.g. for the page https://en.wikipedia.org/wiki/Artificial_intelligence, enter "Artificial_intelligence").
Enter the question you want to ask about the topic.
The script will then generate a summary of the Wikipedia page and extract the answer to your question using a question-answering algorithm.

Functions
scrape_wikipedia_pages(topic)
This function takes a Wikipedia page topic as input and returns the text content of the page.

preprocess_text(text)
This function takes a block of text as input, preprocesses the text data and returns the preprocessed text.

divide_documents(text)
This function takes preprocessed text as input and divides it into smaller parts (paragraphs and sentences).

generate_summaries(sentences)
This function takes a list of sentences as input and generates summaries for each part of the document using a summarization algorithm.

answer_questions(summaries, question)
This function takes a list of summaries and a question as input and uses a Question-Answering algorithm to extract the answers from the summarized data for the given question.

Example
Here is an example usage of the script:

python
Copy code
number = int(input("Enter the number of topics: "))
for i in range(number):
    topic=''
    text=''
    topic = input("Enter the topic: ")
    text = scrape_wikipedia_pages(topic)
text = preprocess_text(text)
sentences = divide_documents(text)
summaries = generate_summaries(sentences)
question = input("Enter the question: ")
answers = answer_questions(summaries, question)
print(question)
print("Answer: " ,answers)
In this example, the script asks the user for the number of topics they want to summarize and ask questions about. For each topic, the user is prompted to enter the name of the topic as it appears in the Wikipedia URL. The script then generates a summary of the Wikipedia page and prompts the user to enter a question. The script then extracts the answer to the question using a Question-Answering algorithm and prints it to the console.
 
 
 
 Heroku Server requires a payment info to let us create a server, but my payment credentials are not working. So due to unexpected difficulty i could not deploy flask server into it.
