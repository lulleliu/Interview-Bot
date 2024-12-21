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

from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredPDFLoader

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

import bs4
import re

import openai
from dotenv import load_dotenv
import os

from typing_extensions import override
from openai import AssistantEventHandler

import requests
import random


load_dotenv()

client = openai.OpenAI()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# print("key: " + openai_api_key)

# parse PDF
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(
    input_files=["cases.pdf"], file_extractor=file_extractor
).load_data()
# print(documents)

full_text = "\n".join([doc.get_content() for doc in documents])
# print(full_text)

# split cases according to ALL CAPS 
pattern = r"(?=^[A-Z0-9 :,#]+$)"
cases_list = re.split(pattern, full_text, flags=re.MULTILINE)
cases_list = [c.strip() for c in cases_list if c.strip()]

# convert each case into langchain.schema.Document
case_docs = []
for case_text in cases_list:
    lines = case_text.split("\n", 1)
    title = lines[0].strip() if lines else "UNKNOWN"
    body = lines[1].strip() if len(lines) > 1 else ""
    # use title as metadata
    case_docs.append(Document(page_content=case_text, metadata={"title": title}))


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=case_docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

def documents_to_text(docs):
    return "\n\n".join([re.sub(r"\s+", " ", d.page_content) for d in docs])


prompt_template = """You are an interviewer for an interview in the consulting industry. You can formulate concise and clear questions from the following retrieved case and user input:

-----
{0}
-----
"""


def chat_with_openai(user_input, history=[]):
    # get relevant documents via retriever
    messages_for_retriever = history[-2:]
    messages_for_retriever.append({"role": "user", "content": user_input})
    user_input_rag = " ".join([i["content"] for i in messages_for_retriever])
    print("USER INPUT RAG: " + user_input_rag)
    docs = retriever.get_relevant_documents(
        user_input_rag
    )  # Now we use 2 of the users messages and one of the chatbots messages to get the relevant documents
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
        messages=messages_list,
        # to add temperature setting?
    )

    # Return the chatbot's reply
    return completion.choices[0].message.content, history


def get_posts():
    # Define the API endpoint URL
    url = "http://localhost:54321/furhat/"

    try:
        # Make a GET request to the API endpoint using requests.get()
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            posts = response.json()
            return posts
        else:
            print("in else")
            print("Error:", response.status_code)
            return None

    except requests.exceptions.RequestException as e:
        # Handle any network-related errors or exceptions
        print("in except")
        print("Error:", e)
        return None


def furhat_say(text_to_say):
    # Define the API base URL
    BASE_URL = "http://localhost:54321/furhat/say"

    # Prepare the parameters
    params = {
        "text": text_to_say,
        "blocking": True,  # Optional: Wait for the speech to finish
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
    params = {"language": language, "blocking": True}

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
    print("👋 Welcome to the Consulting Case Interview! Type 'exit' to end the chat.")
    print(
        "We will walk through 3 different case interviews. Type 'move on' or 'next' to go to the next case.\n"
    )

    # randomly select 3 cases from case_docs
    if len(case_docs) < 3:
        print("Not enough cases in PDF to run a 3-case interview.")
        return
    selected_cases = random.sample(case_docs, 3)

    # current case index
    current_case_idx = 0
    history = []

    greeting = "Hello! Let's begin your case interview now. We have 3 cases lined up."
    furhat_say(greeting)
    history.append({"role": "assistant", "content": greeting})

    while True:
        user_input = furhat_listen("en-US")

        # exit
        if user_input["message"].lower() == "exit":
            print("Bot: Thank you for your time. The interview session has ended.\n")
            break

        # case switch
        if user_input["message"].lower() in ["move on", "next"]:
            current_case_idx += 1
            if current_case_idx >= 3:
                # all 3 cases finished
                closing_msg = "We have finished the case interview session. Thank you for interviewing! Goodbye."
                furhat_say(closing_msg)
                print(f"Bot: {closing_msg}\n")
                break
            else:
                # move on to next case
                case_transition_msg = (
                    f"Alright, let's move on to case #{current_case_idx + 1}."
                )
                furhat_say(case_transition_msg)
                print(f"Bot: {case_transition_msg}\n")
                history.append({"role": "assistant", "content": case_transition_msg})
                continue

        case_context = selected_cases[current_case_idx].page_content
        user_input_rag = f"{user_input}\nRelevant Case Content:\n{case_context}"

        response, history = chat_with_openai(user_input_rag, history)
        history.append({"role": "assistant", "content": response})

        furhat_say(response)


if __name__ == "__main__":
    start_chatbot()
