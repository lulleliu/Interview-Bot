import openai
from dotenv import load_dotenv
import os

from typing_extensions import override
from openai import AssistantEventHandler

import requests
 

load_dotenv()

#openai_api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key = openai_api_key
#print("key: " + openai_api_key)


<<<<<<< Updated upstream
my_api_key=os.environ.get("OPENAI_API_KEY")
print(my_api_key)
=======
persist_directory = "./rag_data/data"

# Load the document from a website
loader = WebBaseLoader(
    web_path="https://en.wikipedia.org/wiki/2024_United_States_presidential_election"
)
docs = loader.load()
>>>>>>> Stashed changes


<<<<<<< Updated upstream
client = openai.OpenAI(api_key=my_api_key)
=======
# Embed the parts and put them in a vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore_test = Chroma(
    collection_name="my_collection",  # Name of your collection
    embedding_function=embeddings,
    persist_directory=persist_directory)

# Add your data to the vectorstore (if needed)
texts = ["This is a sample text.", "Another piece of text."]
metadata = [{"source": "doc1"}, {"source": "doc2"}]
vectorstore_test.add_texts(texts=texts, metadatas=metadata)

stored_data = vectorstore_test._collection.get()

print(f"Data stored at: {persist_directory}")
print("Stored Data:")
print(stored_data)

vectorstore_test.reset_collection()

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
print("Vectorstore Data:")
print(vectorstore._collection.get("embeddings"))


retriever = vectorstore.as_retriever()
>>>>>>> Stashed changes

def chat_with_openai(user_input):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Use the GPT-4o-mini model
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Provide concise answers. Also, you are are speaking so adapt your responses to spoken language. "},  # System message
            {"role": "user", "content": user_input},  # User input
        ]
    )

    # Return the chatbot's reply
    return completion.choices[0].message.content

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

    # Implement a starting interface, where user gets to input language, type of interview etc. And provide information about the role.

    while True:
        user_input = furhat_listen("en-US")
        # user_input = input("You: ")

        if user_input["message"].lower() == 'exit':
            print("Goodbye! 👋")
            break

        response = chat_with_openai(user_input["message"])
        furhat_say(response)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    start_chatbot()
