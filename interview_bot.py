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

"""
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

print(case_docs)
"""

case_docs = []
case_docs.append(Document(page_content="# CABLE TELEVISION COMPANY\n\nQ: Your client is a small holding company that owns three cable television companies in the Northeast: Rochester, NY, Philadelphia and Stamford, CT. Each of these three companies is profitable, and each has been experiencing steadily growing sales over the past few years. However, the management feels that the Northeast is not the fastest growing area of the country, and, therefore, acquired another cable television company in Tucson, Arizona a little over a year ago. Despite every effort of management, the Tucson company’s sales have been stagnant, and the company has been losing money. How would you analyze this situation, and what could be the cause of the poor performance of the Tucson cable company?\n\n# To be divulged gradually:\n\nThe Tucson area is smaller than Philadelphia, but larger than Rochester and Stamford. Tucson is also growing at 12% per year on average. Per capita income is higher than in Philadelphia and the same as in Rochester and in Stamford.\n\nOperating costs in Tucson are essentially the same as in the other markets. The cost of programming is based on number of subscribers and is equal across the nation. Operating costs are composed of variable items: sales staff, maintenance, administration and marketing. Only maintenance is higher than in the other markets, due to the larger land area serviced. Fixed costs relate to the cable lines, which is a function of physical area covered.\n\nThe Tucson company has attempted marketing efforts in the past, such as free Disney programming for one month, free HBO for one month, free hookup, etc. These programs have been modeled after the other three markets.\n\nCable penetration rates in the three Northeastern markets average 45%. The penetration rate in Tucson is 20%. These rates have been steady over the past three years in the Northeast. The penetration rate in Tucson has only risen by 2% in the past three years in Tucson.\n\nThere is only one real substitute good for cable television: satellite dishes. However, many communities are enacting legislation that limits their usage in Tucson. They are also prohibitively expensive for most people.\n\n# Solution:\n\nThe real error of management results from their failure to recognize another “substitute” good: no cable television at all; television reception is far better in the desert Southwest than in Northeastern cities. The lower penetration rate is most likely a result of different climate conditions and lower interference in Arizona.", metadata={"title": "# CABLE TELEVISION COMPANY" }))
case_docs.append(Document(page_content="# FRENCH PIZZA MARKET\n\nPizza Hut has recently entered the home pizza delivery business in Paris. The market for home delivery is currently dominated by Spizza Pizza. Pizza Hut has asked your consulting firm to help it analyze issues that will determine its likelihood of success in the Parisian Pizza market. First, what information would you need and second, how would you analyze the pizza delivery market?\n\n# Possible Information Needs:\n\nAn estimate of the size of the Parisian home pizza delivery market. This could be obtained by knowing the population of Paris (6 million) and making some educated guesses about factors that determine pizza market size.\n\nYou may also want to know the size of Spizza, the current competitor, including sales, number of stores, and proportion of Paris that is currently served by Spizza.\n\nOther useful information: market segments targeted and served by Spizza; market segments that are neglected by Spizza; what type of product do they offer; what do they charge for their product; what is the cost structure of their business and what products are most profitable.\n\n# Method of analysis:\n# The best method of analysis\n\nThe best method of analysis would start by determining if any part of the market is not well served currently by Spizza. Determine what are the needs of any neglected market, and understand if your client could profitably serve this market.\n\nAlso, try to understand the likely competitive response of Spizza to your client’s entry. How will you defend your position if Spizza decides to fight for market share?", metadata={"title" : "# FRENCH PIZZA MARKET"}))
case_docs.append(Document(page_content="# LOCAL BANKING DEMAND\n\nHow would you determine whether a location in New York City holds enough banking demand to warrant opening a branch?\n\n# Suggested framework:\n\nBecause this is a demand-oriented question, one should consider a marketing framework, such as the 4 P’s.\n\n# Interviewer Notes:\n\nThe demographics of the area surrounding the prospective branch should be examined. Population, business concentration, income levels, etc. should be compared with those of historically successful branches.\n\nCompetitor reactions could easily make this venture unprofitable, so it is essential to anticipate them. These will depend on the importance of the area to competitors (in terms of profit, share, etc.)\n\nThe client will have to match competitors’ incentives to customers and should estimate the cost of doing so.\n\nThe client must examine if the new branch would complement their existing competence and strategy (retail or commercial, high growth or high profitability, etc.) and what purpose it would serve. If the need focuses on deposits and withdrawals only, maybe a cash machine would suffice.", metadata={"title": "# LOCAL BANKING DEMAND"}))

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

def get_relevant_case(description):
    relevant_case = retriever.get_relevant_documents(description)
    return relevant_case

def chat_with_openai(user_input, history=[]):
    # get relevant documents via retriever
    messages_for_retriever = history[-2:]
    messages_for_retriever.append({"role": "user", "content": user_input})

    user_input_rag = " ".join([i["content"] for i in messages_for_retriever])

    # print("USER INPUT RAG: " + user_input_rag)

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

    print("-------------------MESSAGES LIST--------------------- \n")
    for i in messages_list:
        print(str(i) + "\n")

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
    user_preferences = []
    user_preferences.append(input("What type of case would you like to practice? For example, market sizing, profitability cases, pricing cases, etc."))
    relevant_case = get_relevant_case(user_preferences[0])
    print(relevant_case)
    # print(
    #     "We will walk through 3 different case interviews. Type 'move on' or 'next' to go to the next case.\n"
    # )

    # randomly select 3 cases from case_docs
    if len(case_docs) < 3:
        print("Not enough cases in PDF to run a 3-case interview.")
        return
    
    selected_cases = random.sample(case_docs, 3)

    # current case index
    current_case_idx = 0
    history = []

    greeting = "Hello! Let's begin your case interview now. We have 3 cases lined up."
    #furhat_say(greeting)
    history.append({"role": "assistant", "content": greeting})

    while True:
        #user_input = furhat_listen("en-US")
        user_input = input("You: ")

        # exit
        #if user_input["message"].lower() == "exit":
        if user_input.lower() == "exit":
            print("Bot: Thank you for your time. The interview session has ended.\n")
            break

        # case switch
        #if user_input["message"].lower() in ["move on", "next"]:
        if user_input.lower() in ["move on", "next"]:
            current_case_idx += 1
            if current_case_idx >= 3:
                # all 3 cases finished
                closing_msg = "We have finished the case interview session. Thank you for interviewing! Goodbye."

                #furhat_say(closing_msg)

                print(f"Bot: {closing_msg}\n")
                break

            else:
                # move on to next case
                case_transition_msg = (
                    f"Alright, let's move on to case #{current_case_idx + 1}."
                )

                #furhat_say(case_transition_msg)

                print(f"Bot: {case_transition_msg}\n")

                history.append({"role": "assistant", "content": case_transition_msg})

                continue

        ##### NEW!
        #case_context = selected_cases[current_case_idx].page_content
        case_context = relevant_case
        

        user_input_rag = f"{user_input}\nRelevant Case Content:\n{case_context}"

        response, history = chat_with_openai(user_input_rag, history)

        history.append({"role": "assistant", "content": response})

        #furhat_say(response)
        print("Bot: " + response)


if __name__ == "__main__":
    start_chatbot()
