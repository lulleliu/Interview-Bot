import tkinter as tk
import json
import re
import random
import os
import threading
import speech_recognition as sr

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

from llm_as_a_judge import judge_single_answer
from llm_as_a_judge import get_judge_response

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

client = openai.OpenAI()

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
You can formulate concise and clear questions, but try to not make them too guiding initially, from the following retrieved case context and user input:

-----
{0}
-----

When speaking to the user, attemt to limit it to one question at a time. Instead use multiple rounds of dialogue to ask all of your questions.
Challenge the user on some of their assumptions and make sure they can motivate their answers somehow. However, when you feel that a proper solution to the case has been given you can tell that to the user and follow up with: "Is there anything else you would like to add or are wondering?" or something like it. In the end, once everything is finished, tell the user to write 'exit' to finish the program and get their results.
If you feel like you are asking the same question over and over, just continue and disregard that question.
Also, only answer in a format which is readable to humans.

"""

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

###################################################
# 3. Main chat logic: RAG + Chat
###################################################

def get_relevant_case(description):
    relevant_case = retriever.get_relevant_documents(description)
    return relevant_case

def chat_with_openai(user_input, history=[]):
    history.append({"role": "user", "content": user_input})

    print("-------------------MESSAGES LIST--------------------- \n")
    for i in history:
        print(str(i) + "\n")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Use the GPT-4o-mini model
        messages=history,
        # to add temperature setting?
    )

    # Return the chatbot's reply
    return completion.choices[0].message.content, history


# def chat_with_openai(user_input, history):
#     """
#     history: A list of dictionaries with keys "role" and "content", e.g.:
#              [{"role": "system", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

#     Steps:
#     1) Merge user_input (and optionally recent conversation) into a single query for the retriever.
#     2) Use the retriever to fetch the most relevant document => system prompt context.
#     3) Construct the message list for OpenAI, with the retrieved text in the system prompt.
#     4) Call OpenAI ChatCompletion to get the response.
#     5) Append the assistant response to history and return it.
#     """
#     # 1) Prepare input for retrieval
#     user_input_rag = user_input

#     # 2) Retrieve documents
#     docs = retriever.get_relevant_documents(user_input_rag)
#     doc_text = documents_to_text(docs)
#     system_prompt = prompt_template.format(doc_text)
#     print(system_prompt)

#     # 3) Build the messages list
#     #    - system_prompt at the beginning
#     messages_list = [{"role": "system", "content": system_prompt}] + history
#     #    - add the new user message
#     messages_list.append({"role": "user", "content": user_input})

#     # Update local history
#     history.append({"role": "user", "content": user_input})

#     # 4) Call OpenAI
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages_list
#     )
#     response = completion.choices[0].message.content

#     # 5) Add assistant response to history
#     history.append({"role": "assistant", "content": response})

#     return response, history


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
        self.case_display = tk.Text(master, wrap=tk.WORD, height=10, width=70)
        self.case_display.pack(pady=5)

        # Text widget: conversation display
        self.chat_display = tk.Text(master, wrap=tk.WORD, height=15, width=70)
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
        self.send_button = tk.Button(master, text="Send", command=lambda: self.on_send(False))
        self.send_button.pack(side=tk.LEFT, padx=5)

        # Add speech-to-text functionality

        self.speech_button = tk.Button(master, text="Speech-to-Text", command=self.toggle_listen)
        #self.speech_button = tk.Button(master, text="Speech-to-Text", command=furhat_listen("en-US"))
        self.speech_button.pack(side=tk.LEFT, padx=5)

        # Conversation state
        self.history = []
        self.current_case_idx = 0
        self.is_listening = False

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

    def on_send(self, STT):
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
        if not STT:
            self.chat_display.insert(tk.END, f"You: {user_input}\n", "user")

        # Handle special commands
        if user_input.lower() == "exit":
            self.chat_display.insert(tk.END, "Bot: Thank you. The session has ended.\n Please Wait for your score, it will be displayed in the chat below.\n", "bot")
            # furhat_say("Thank you. The session has ended. Please Wait for your score, it will be displayed in the chat below.")
            interview = ""
            # Save Interview as string in JSON format
            for i in self.history:
                if i["content"] != "you are an interviewer, please begin with asking me to tell you about myself and why i am interested in a career in consulting. Then after my response proceed with introducing a case.":
                    interview += f'{{"role": "{i["role"]}", "content": "{i["content"]}"}}'
                    interview += ",\n"
            judge_res = get_judge_response(interview)
            self.chat_display.insert(tk.END, judge_res)
            self.send_button.config(state="disabled")
            self.speech_button.config(state="disabled")
            with open("interview_session_no_furhat.txt", "w", encoding="utf-8") as file:
                file.write(interview)


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

            docs = get_relevant_case(user_input)
            if docs:
                self.case_display.insert(tk.END, "[Relevant Case Found]\n" + docs[0].page_content + "\n\n", "bot")

            # Remove "thinking" line if you want
            # For demonstration, we'll just keep it and add the next line
            doc_text = documents_to_text(docs) if docs else ""
            system_prompt = prompt_template.format(doc_text)
            self.history.append({"role": "system", "content": system_prompt})

            welcome_msg = (
                "I'm here to help with your chosen case topic. A relevant case for your topic is displayed in the upper text window.\n"
                "Please make sure that you are speaking clearly. For the best speech to text results, after clicking the Speech-to-text button please wait a second before speaking. Also please wait a second before stopping the recording. \n"
                "Okay, lets begin. Please tell me about yourself and why you are interested in consulting. "
            )
            # Insert the final bot message in blue
            self.chat_display.insert(tk.END, f"Bot: {welcome_msg}\n", "bot")

            # furhat_say(welcome_msg)

            self.history.append({"role": "assistant", "content": welcome_msg})
            return
        
        # ============ STEP A: JUDGE USER'S ANSWER & GESTURE ============
        # We do this BEFORE the bot crafts its answer.
        rating, explanation = judge_single_answer(user_input)
        self.chat_display.insert(
            tk.END, f"[Judge] Rating: {rating}, Explanation: {explanation}\n", "thinking"
        )

        # ============ STEP B: BOT (RAG + Chat) ============

        # For subsequent inputs, show "thinking..."
        self.chat_display.insert(tk.END, "Bot is thinking...\n", "thinking")
        self.chat_display.update_idletasks()

        # Then perform normal RAG + Chat
        response, self.history = chat_with_openai(user_input, self.history)
        self.history.append({"role": "assistant", "content": response})


        # Insert the final bot response in blue
        self.chat_display.insert(tk.END, f"Bot: {response}\n", "bot")
        # furhat_say(response)
        

    def toggle_listen(self):
        if self.is_listening:
            # Stop listening and process the collected audio
            self.is_listening = False
            self.speech_button.config(text="Speech-to-Text")
            self.chat_display.insert(tk.END, "Processing all recorded audio...\n", "thinking")
            self.process_audio()
            self.on_send(True)
        else:
            # Start listening continuously
            self.is_listening = True
            self.speech_button.config(text="Stop Listening")
            threading.Thread(target=self.listen).start()  # Run listening in a separate thread

    def listen(self):
        """
        Continuously listen for audio until the user clicks "Stop Listening."
        All audio is saved to a buffer for later processing.
        """
        self.recognizer = sr.Recognizer()
        self.audio_buffer = []  # Buffer to store all audio chunks

        with sr.Microphone() as source:
            self.chat_display.insert(tk.END, "Adjusting for background noise...\n", "thinking")
            self.recognizer.adjust_for_ambient_noise(source)

            while self.is_listening:
                try:
                    self.chat_display.insert(tk.END, "Listening...\n", "thinking")
                    audio = self.recognizer.listen(source, timeout=None)  # Listen indefinitely
                    self.audio_buffer.append(audio)  # Store audio in the buffer
                except Exception as e:
                    self.chat_display.insert(tk.END, f"Listening error: {e}\n", "bot")
                    break

                self.chat_display.see(tk.END)  # Scroll to the end

    def process_audio(self):
        """
        Processes all the audio chunks stored in the buffer.
        This is called when the user clicks "Stop Listening."
        """
        if not self.audio_buffer:
            self.chat_display.insert(tk.END, "No audio recorded.\n", "bot")
            return

        full_text = []
        for i, audio in enumerate(self.audio_buffer):
            try:
                # self.chat_display.insert(tk.END, f"Processing audio chunk {i + 1}...\n", "thinking")
                text = self.recognizer.recognize_google(audio)
                full_text.append(text)
            except sr.UnknownValueError:
                self.chat_display.insert(tk.END, f"Could not understand audio chunk {i + 1}.\n", "bot")
            except sr.RequestError as e:
                self.chat_display.insert(tk.END, f"Service error for chunk {i + 1}: {e}\n", "bot")

        # Combine all recognized text and display
        combined_text = " ".join(full_text)
        self.chat_display.insert(tk.END, f"You said (via STT): {combined_text}\n", "user")
        self.input_field.insert(0, combined_text)  # Pre-fill the input field with recognized text




def main():
    root = tk.Tk()
    app = InterviewBotApp(root)
    # If you want to display the first case content in the UI immediately, you can:
    # first_case_content = app.selected_cases[0].page_content
    # app.chat_display.insert(tk.END, "[Case Content]\n" + first_case_content + "\n\n")
    root.mainloop()

if __name__ == "__main__":
    main()
