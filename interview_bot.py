from jupyter_chat import *
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
import re

import openai
from dotenv import load_dotenv
import os

from typing_extensions import override
from openai import AssistantEventHandler

import requests
 

load_dotenv()

client = openai.OpenAI()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

#print("key: " + openai_api_key)

# === RAG part begins ===

# Load the document from a website
loader = WebBaseLoader(
    web_path="https://en.wikipedia.org/wiki/2024_United_States_presidential_election"
)
docs = loader.load()

# Split the document into parts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed the parts and put them in a vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Define a function that combine a set of document parts into a string, also removing excessive whitespaces
def documents_to_text(docs):
    return "\n\n".join([re.sub(r'\s+', ' ', doc.page_content) for doc in docs])

prompt_template = """You are a helpful assistant. You know the following information:

-----
{0}
-----

"""

# === RAG part ends ===

def chat_with_openai(user_input, history=[]):
    # get relevant documents via retriever
    docs = retriever.get_relevant_documents(user_input) #MODIFY SO THAT THE LATEST 3 MESSAGES ARE USED
    information = documents_to_text(docs)
    # integrate RAG information into system prompt
    system_prompt = prompt_template.format(information)
    
    messages_list = []

    messages_list.append({"role": "system", "content": system_prompt})
    for i in history:
        messages_list.append(i)
    messages_list.append({"role": "user", "content": user_input})
    history.append({"role": "user", "content": user_input})
    print("MESSAGES LIST: " + str(messages_list))
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Use the GPT-4o-mini model
        messages= messages_list
        # to add temperature setting?
    )

    # Return the chatbot's reply
    return completion.choices[0].message.content, history

def get_posts():
    # Define the API endpoint URL
    url = 'http://localhost:54321/furhat/'

    try:
        # Make a GET request to the API endpoint using requests.get()
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            posts = response.json()
            return posts
        else:
            print("in else")
            print('Error:', response.status_code)
            return None
        
    except requests.exceptions.RequestException as e:
        # Handle any network-related errors or exceptions
        print("in except")
        print('Error:', e)
        return None
    
def furhat_say(text_to_say):
    # Define the API base URL
    BASE_URL = "http://localhost:54321/furhat/say"

    # Prepare the parameters
    params = {
        "text": text_to_say,
        "blocking": True  # Optional: Wait for the speech to finish
    }

    try:
        # Make the POST request
        response = requests.post(BASE_URL, params=params)

        # Check the response
        if response.status_code == 200:
            print("Furhat spoke successfully.")
            print("Response:", response.json())
        else:
            print(f"Failed to make Furhat speak. Status Code: {response.status_code}")
            print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)

def furhat_listen(language):
    # Define the API base URL
    BASE_URL = "http://localhost:54321/furhat/listen"

    # Prepare the parameters
    params = {
        "language": language,
        "blocking": True
    }

    try:
        # Make the Get request
        response = requests.get(BASE_URL, params=params)

        # Check the response
        if response.status_code == 200:
            print("Furhat listened successfully.")
            print("Response:", response.json())
            return response.json()
        else:
            print(f"Failed to make Furhat listen. Status Code: {response.status_code}")
            print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
    

def start_chatbot():

    print("👋 Welcome! I'm your chatbot. Type 'exit' to end the chat.\n")
    history = []
    while True:
        #user_input = furhat_listen("en-US")
        user_input = input("You: ")

        #if user_input["message"].lower() == 'exit':
        if user_input.lower() == 'exit':
            print("Goodbye! 👋")
            break

        #response = chat_with_openai(user_input["message"])
        
        response, history = chat_with_openai(user_input, history)
        history.append({"role": "assistant", "content": response})
        # furhat_say(response)
        print(f"Bot: {response}\n")



if __name__ == "__main__":
    start_chatbot()
    