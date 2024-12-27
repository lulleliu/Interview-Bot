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
import speech_recognition as sr  # For speech-to-text

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

###################################################
# 1. Load case_docs from JSON & build a retriever
###################################################

def load_case_docs_from_json(json_path="case_docs.json"):
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
    return "\n\n".join([re.sub(r"\s+", " ", d.page_content) for d in docs])

prompt_template = """You are an interviewer for a consulting industry case interview.
You can formulate concise and clear questions from the following retrieved case context and user input:

-----
{0}
-----
"""

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

client = openai.OpenAI()

###################################################
# 3. Main chat logic: RAG + Chat
###################################################

def chat_with_openai(user_input, history):
    user_input_rag = user_input
    docs = retriever.get_relevant_documents(user_input_rag)
    doc_text = documents_to_text(docs)
    system_prompt = prompt_template.format(doc_text)
    messages_list = [{"role": "system", "content": system_prompt}] + history
    messages_list.append({"role": "user", "content": user_input})
    history.append({"role": "user", "content": user_input})
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_list
    )
    response = completion.choices[0].message.content
    history.append({"role": "assistant", "content": response})
    return response, history

###################################################
# 4. Tkinter UI with speech-to-text functionality
###################################################

class InterviewBotApp:
    def __init__(self, master):
        self.master = master
        master.title("Consulting Case Interview Bot")

        self.case_display = tk.Text(master, wrap=tk.WORD, height=20, width=70)
        self.case_display.pack(pady=5)

        self.chat_display = tk.Text(master, wrap=tk.WORD, height=20, width=70)
        self.chat_display.pack(pady=5)
        self.chat_display.tag_configure("user", foreground="green")
        self.chat_display.tag_configure("bot", foreground="blue")
        self.chat_display.tag_configure("thinking", foreground="gray", font=("Helvetica", 9, "italic"))

        self.input_field = tk.Entry(master, width=50)
        self.input_field.pack(side=tk.LEFT, padx=5, pady=5)

        self.send_button = tk.Button(master, text="Send", command=self.on_send)
        self.send_button.pack(side=tk.LEFT, padx=5)

        self.speech_button = tk.Button(master, text="Start Speech-to-Text", command=self.start_speech_to_text)
        self.speech_button.pack(side=tk.LEFT, padx=5)

        self.history = []
        self.current_case_idx = 0

        initial_message = (
            "Bot: Welcome to the Consulting Case Interview! Type 'exit' to end.\n"
            "What type of case would you like to practice?\n"
            "(e.g. market sizing, profitability, etc.)\n"
        )
        self.chat_display.insert(tk.END, initial_message, "bot")
        self.selected_cases = random.sample(case_docs, min(len(case_docs), 3))

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def on_send(self):
        user_input = self.input_field.get().strip()
        if not user_input:
            return
        self.input_field.delete(0, tk.END)
        self.chat_display.insert(tk.END, f"You: {user_input}\n", "user")
        if user_input.lower() == "exit":
            self.chat_display.insert(tk.END, "Bot: Thank you. The session has ended.\n", "bot")
            return
        self.chat_display.insert(tk.END, "Bot is thinking...\n", "thinking")
        self.chat_display.update_idletasks()
        response, self.history = chat_with_openai(user_input, self.history)
        self.chat_display.insert(tk.END, f"Bot: {response}\n", "bot")

    def start_speech_to_text(self):
        try:
            with self.microphone as source:
                self.chat_display.insert(tk.END, "Listening... Speak now.\n", "bot")
                audio = self.recognizer.listen(source)
            self.chat_display.insert(tk.END, "Processing speech...\n", "bot")
            user_input = self.recognizer.recognize_google(audio)
            self.chat_display.insert(tk.END, f"You (via speech): {user_input}\n", "user")
            response, self.history = chat_with_openai(user_input, self.history)
            self.chat_display.insert(tk.END, f"Bot: {response}\n", "bot")
        except sr.UnknownValueError:
            self.chat_display.insert(tk.END, "Bot: Sorry, I couldn't understand that. Please try again.\n", "bot")
        except sr.RequestError as e:
            self.chat_display.insert(tk.END, f"Bot: Could not request results from Google Speech Recognition; {e}\n", "bot")

def main():
    root = tk.Tk()
    app = InterviewBotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
