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


my_api_key=os.environ.get("OPENAI_API_KEY")
print(my_api_key)


client = openai.OpenAI(api_key=my_api_key)

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
