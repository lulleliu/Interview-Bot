# import tkinter as tk
# import json
# import re
# import random
# import os

# from dotenv import load_dotenv
# load_dotenv()

# import openai
# from openai import AssistantEventHandler
# from typing_extensions import override
# import requests
# import speech_recognition as sr  # For speech-to-text

# from langchain.schema import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# from llama_parse import LlamaParse
# from llama_index.core import SimpleDirectoryReader

# ###################################################
# # 1. Load case_docs from JSON & build a retriever
# ###################################################

# def load_case_docs_from_json(json_path="case_docs.json"):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     loaded_docs = []
#     for item in data:
#         page_content = item["page_content"]
#         metadata = item["metadata"]
#         doc = Document(page_content=page_content, metadata=metadata)
#         loaded_docs.append(doc)
#     return loaded_docs

# case_docs = load_case_docs_from_json("case_docs.json")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma.from_documents(documents=case_docs, embedding=embeddings)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# ###################################################
# # 2. Helper functions
# ###################################################

# def documents_to_text(docs):
#     return "\n\n".join([re.sub(r"\s+", " ", d.page_content) for d in docs])

# prompt_template = """You are an interviewer for a consulting industry case interview.
# You can formulate concise and clear questions from the following retrieved case context and user input:

# -----
# {0}
# -----
# """

# openai_api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = openai_api_key

# client = openai.OpenAI()

# ###################################################
# # 3. Main chat logic: RAG + Chat
# ###################################################

# def chat_with_openai(user_input, history):
#     user_input_rag = user_input
#     docs = retriever.get_relevant_documents(user_input_rag)
#     doc_text = documents_to_text(docs)
#     system_prompt = prompt_template.format(doc_text)
#     messages_list = [{"role": "system", "content": system_prompt}] + history
#     messages_list.append({"role": "user", "content": user_input})
#     history.append({"role": "user", "content": user_input})
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages_list
#     )
#     response = completion.choices[0].message.content
#     history.append({"role": "assistant", "content": response})
#     return response, history

# ###################################################
# # 4. Tkinter UI with speech-to-text functionality
# ###################################################

# class InterviewBotApp:
#     def __init__(self, master):
#         self.master = master
#         master.title("Consulting Case Interview Bot")

#         self.case_display = tk.Text(master, wrap=tk.WORD, height=20, width=70)
#         self.case_display.pack(pady=5)

#         self.chat_display = tk.Text(master, wrap=tk.WORD, height=20, width=70)
#         self.chat_display.pack(pady=5)
#         self.chat_display.tag_configure("user", foreground="green")
#         self.chat_display.tag_configure("bot", foreground="blue")
#         self.chat_display.tag_configure("thinking", foreground="gray", font=("Helvetica", 9, "italic"))

#         self.input_field = tk.Entry(master, width=50)
#         self.input_field.pack(side=tk.LEFT, padx=5, pady=5)

#         self.send_button = tk.Button(master, text="Send", command=self.on_send)
#         self.send_button.pack(side=tk.LEFT, padx=5)

#         self.speech_button = tk.Button(master, text="Start Speech-to-Text", command=self.start_speech_to_text)
#         self.speech_button.pack(side=tk.LEFT, padx=5)

#         self.history = []
#         self.current_case_idx = 0

#         initial_message = (
#             "Bot: Welcome to the Consulting Case Interview! Type 'exit' to end.\n"
#             "What type of case would you like to practice?\n"
#             "(e.g. market sizing, profitability, etc.)\n"
#         )
#         self.chat_display.insert(tk.END, initial_message, "bot")
#         self.selected_cases = random.sample(case_docs, min(len(case_docs), 3))

#         self.recognizer = sr.Recognizer()
#         self.microphone = sr.Microphone()

#     def on_send(self):
#         user_input = self.input_field.get().strip()
#         if not user_input:
#             return
#         self.input_field.delete(0, tk.END)
#         self.chat_display.insert(tk.END, f"You: {user_input}\n", "user")
#         if user_input.lower() == "exit":
#             self.chat_display.insert(tk.END, "Bot: Thank you. The session has ended.\n", "bot")
#             return
#         self.chat_display.insert(tk.END, "Bot is thinking...\n", "thinking")
#         self.chat_display.update_idletasks()
#         response, self.history = chat_with_openai(user_input, self.history)
#         self.chat_display.insert(tk.END, f"Bot: {response}\n", "bot")

#     def start_speech_to_text(self):
#         try:
#             with self.microphone as source:
#                 self.chat_display.insert(tk.END, "Listening... Speak now.\n", "bot")
#                 audio = self.recognizer.listen(source)
#             self.chat_display.insert(tk.END, "Processing speech...\n", "bot")
#             user_input = self.recognizer.recognize_google(audio)
#             self.chat_display.insert(tk.END, f"You (via speech): {user_input}\n", "user")
#             response, self.history = chat_with_openai(user_input, self.history)
#             self.chat_display.insert(tk.END, f"Bot: {response}\n", "bot")
#         except sr.UnknownValueError:
#             self.chat_display.insert(tk.END, "Bot: Sorry, I couldn't understand that. Please try again.\n", "bot")
#         except sr.RequestError as e:
#             self.chat_display.insert(tk.END, f"Bot: Could not request results from Google Speech Recognition; {e}\n", "bot")

# def main():
#     root = tk.Tk()
#     app = InterviewBotApp(root)
#     root.mainloop()

# if __name__ == "__main__":
#     main()




import tkinter as tk
import json
import re
import random
import os

from dotenv import load_dotenv
load_dotenv()

import openai
from openai import AssistantEventHandler
from typing_extensions import override
import requests

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

###################################################
# 1. Load case_docs from JSON & build a retriever
###################################################

def load_case_docs_from_json(json_path="case_docs.json"):
    """
    Reads a JSON file that contains the parsed documents and reconstructs
    them into a list of langchain.schema.Document objects.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    loaded_docs = []
    for item in data:
        page_content = item["page_content"]
        metadata = item["metadata"]
        doc = Document(page_content=page_content, metadata=metadata)
        loaded_docs.append(doc)
    return loaded_docs

case_docs = load_case_docs_from_json("case_docs.json")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=case_docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

###################################################
# 2. Helper functions
###################################################

def documents_to_text(docs):
    """
    Joins the content of a list of Document objects into a single string,
    replacing consecutive whitespace with a single space.
    """
    return "\n\n".join([re.sub(r"\s+", " ", d.page_content) for d in docs])

# This prompt template is used in the system prompt to incorporate retrieved context.
prompt_template = """You are an interviewer for a consulting industry case interview.
You can formulate concise and clear questions from the following retrieved case context and user input:

-----
{0}
-----
"""

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

client = openai.OpenAI()  # Used for client.chat.completions.create(...)

###################################################
# 3. Main chat logic: RAG + Chat
###################################################

def chat_with_openai(user_input, history):
    """
    history: A list of dictionaries with keys "role" and "content", e.g.:
             [{"role": "system", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Steps:
    1) Merge user_input (and optionally recent conversation) into a single query for the retriever.
    2) Use the retriever to fetch the most relevant document => system prompt context.
    3) Construct the message list for OpenAI, with the retrieved text in the system prompt.
    4) Call OpenAI ChatCompletion to get the response.
    5) Append the assistant response to history and return it.
    """
    # 1) Prepare input for retrieval
    user_input_rag = user_input

    # 2) Retrieve documents
    docs = retriever.get_relevant_documents(user_input_rag)
    doc_text = documents_to_text(docs)
    system_prompt = prompt_template.format(doc_text)

    # 3) Build the messages list
    #    - system_prompt at the beginning
    messages_list = [{"role": "system", "content": system_prompt}] + history
    #    - add the new user message
    messages_list.append({"role": "user", "content": user_input})

    # Update local history
    history.append({"role": "user", "content": user_input})

    # 4) Call OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_list
    )
    response = completion.choices[0].message.content

    # 5) Add assistant response to history
    history.append({"role": "assistant", "content": response})

    return response, history

###################################################
# 4. Tkinter UI to display and handle conversation
###################################################

class InterviewBotApp:
    def __init__(self, master):
        """
        Creates a main Tk window with:
         - A Text widget for displaying conversation
         - An Entry for user input
         - A Button to send the user input
        """
        self.master = master
        master.title("Consulting Case Interview Bot")

        # Text widget: relevant case display
        self.case_display = tk.Text(master, wrap=tk.WORD, height=20, width=70)
        self.case_display.pack(pady=5)

        # Text widget: conversation display
        self.chat_display = tk.Text(master, wrap=tk.WORD, height=20, width=70)
        self.chat_display.pack(pady=5)
        

        # Configure text tags to style different message roles
        # user = green, bot = blue, thinking = gray (italic)
        self.chat_display.tag_configure("user", foreground="green")
        self.chat_display.tag_configure("bot", foreground="blue")
        self.chat_display.tag_configure("thinking", foreground="gray", font=("Helvetica", 9, "italic"))

        # Entry for user input
        self.input_field = tk.Entry(master, width=50)
        self.input_field.pack(side=tk.LEFT, padx=5, pady=5)

        # Button to send
        self.send_button = tk.Button(master, text="Send", command=self.on_send)
        self.send_button.pack(side=tk.LEFT, padx=5)

        # Conversation state
        self.history = []
        self.current_case_idx = 0

        # Ask the user what type of case they'd like to practice
        initial_message = (
            "Bot: Welcome to the Consulting Case Interview! Type 'exit' to end.\n"
            "What type of case would you like to practice?\n"
            "(e.g. market sizing, profitability, etc.)\n"
        )
        self.chat_display.insert(tk.END, initial_message, "bot")

        # Randomly select 3 cases (if you want multiple cases)
        if len(case_docs) < 3:
            self.selected_cases = case_docs
        else:
            self.selected_cases = random.sample(case_docs, 3)

    def on_send(self):
        """
        Triggered when the user clicks the 'Send' button:
          1. Retrieve the user's input
          2. Display user's input in green
          3. Show a "thinking" line in gray
          4. Retrieve or chat with OpenAI
          5. Replace or follow up with the bot's actual response
        """
        user_input = self.input_field.get().strip()
        if not user_input:
            return

        # Clear the entry
        self.input_field.delete(0, tk.END)

        # Display user input (green)
        self.chat_display.insert(tk.END, f"You: {user_input}\n", "user")

        # Handle special commands
        if user_input.lower() == "exit":
            self.chat_display.insert(tk.END, "Bot: Thank you. The session has ended.\n", "bot")
            return

        if user_input.lower() in ["move on", "next"]:
            self.current_case_idx += 1
            if self.current_case_idx >= len(self.selected_cases):
                self.chat_display.insert(
                    tk.END,
                    "Bot: We have finished all selected cases. Thank you!\n",
                    "bot"
                )
                return
            else:
                transition_msg = f"Bot: Alright, let's move on to case #{self.current_case_idx + 1}.\n"
                self.chat_display.insert(tk.END, transition_msg, "bot")
                next_case_content = self.selected_cases[self.current_case_idx].page_content
                self.chat_display.insert(tk.END, f"\n[Case Content]\n{next_case_content}\n\n", "bot")
                return

        # If this is the user's first real input, retrieve a relevant doc
        # or simply respond with a prompt
        if len(self.history) == 0:
            # Insert "thinking" placeholder
            thinking_tag = "thinking"
            self.chat_display.insert(tk.END, "Bot is thinking...\n", thinking_tag)
            self.chat_display.update_idletasks()

            docs = retriever.get_relevant_documents(user_input)
            if docs:
                self.case_display.insert(tk.END, "[Relevant Case Found]\n" + docs[0].page_content + "\n\n", "bot")

            # Remove "thinking" line if you want
            # For demonstration, we'll just keep it and add the next line
            doc_text = documents_to_text(docs) if docs else ""
            system_prompt = prompt_template.format(doc_text)
            self.history.append({"role": "system", "content": system_prompt})

            welcome_msg = (
                "Sure! I'm here to help with your chosen case topic. "
                "Please tell me about yourself and why you are interested in consulting. "
                "Then we'll proceed with a case question."
            )
            # Insert the final bot message in blue
            self.chat_display.insert(tk.END, f"Bot: {welcome_msg}\n", "bot")

            self.history.append({"role": "assistant", "content": welcome_msg})
            return

        # For subsequent inputs, show "thinking..."
        self.chat_display.insert(tk.END, "Bot is thinking...\n", "thinking")
        self.chat_display.update_idletasks()

        # Then perform normal RAG + Chat
        response, self.history = chat_with_openai(user_input, self.history)

        # Insert the final bot response in blue
        self.chat_display.insert(tk.END, f"Bot: {response}\n", "bot")



def main():
    root = tk.Tk()
    app = InterviewBotApp(root)
    # If you want to display the first case content in the UI immediately, you can:
    # first_case_content = app.selected_cases[0].page_content
    # app.chat_display.insert(tk.END, "[Case Content]\n" + first_case_content + "\n\n")
    root.mainloop()

if __name__ == "__main__":
    main()
