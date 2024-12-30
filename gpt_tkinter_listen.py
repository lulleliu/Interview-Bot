import tkinter as tk
from tkinter import messagebox
import threading
import speech_recognition as sr

class SpeechToTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech-to-Text")
        self.is_listening = False

        self.listen_button = tk.Button(root, text="Listen", command=self.toggle_listen, width=20, height=2)
        self.listen_button.pack(pady=20)

        self.output_text = tk.Text(root, wrap=tk.WORD, width=50, height=10)
        self.output_text.pack(pady=20)

    def toggle_listen(self):
        if self.is_listening:
            self.is_listening = False
            self.listen_button.config(text="Listen")
        else:
            self.is_listening = True
            self.listen_button.config(text="Stop Listening")
            threading.Thread(target=self.listen).start()  # Run listening in a separate thread

    def listen(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.output_text.insert(tk.END, "Adjusting for background noise...\n")
            recognizer.adjust_for_ambient_noise(source)

            while self.is_listening:
                try:
                    self.output_text.insert(tk.END, "Listening...\n")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    self.output_text.insert(tk.END, "Processing audio...\n")
                    text = recognizer.recognize_google(audio)
                    self.output_text.insert(tk.END, f"You said: {text}\n")
                except sr.UnknownValueError:
                    self.output_text.insert(tk.END, "Could not understand the audio.\n")
                except sr.RequestError as e:
                    self.output_text.insert(tk.END, f"Error with the service: {e}\n")
                except Exception as e:
                    self.output_text.insert(tk.END, f"Error: {e}\n")

                self.output_text.see(tk.END)  # Scroll to the end

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()
