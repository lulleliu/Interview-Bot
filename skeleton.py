import openai
import os

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to interact with the GPT model
def gpt_chatbot():
    print("ChatBot: Hello! I'm a chatbot powered by GPT. Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("ChatBot: Goodbye! Have a great day!")
            break
        
        # Get the response from OpenAI's GPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can change this to "gpt-4" if you have access
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        
        # Extract and print the bot's response
        bot_response = response['choices'][0]['message']['content'].strip()
        print(f"ChatBot: {bot_response}")

# Start the chatbot
if __name__ == "__main__":
    gpt_chatbot()
