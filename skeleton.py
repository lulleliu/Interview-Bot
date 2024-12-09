from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

# Step 1: Set up the LLM (e.g., OpenAI)
llm = OpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Step 2: Define a Prompt Template
interview_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="You are a professional job interviewer. Use the context below to evaluate responses to the question: \"{question}\". Provide constructive feedback. Context: {context}"
)

# Step 3: Set up Memory to Maintain Context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 4: Initialize the Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=interview_prompt
)

# Step 5: Define Core Logic for the Bot
def interview_bot(question, user_response):
    """
    Conduct an interview step: ask a question, capture the user's response,
    and provide feedback.
    """
    # Extract chat history for context
    context = memory.load_memory_variables({}).get("chat_history", "")

    # Ensure input matches the prompt's expected input variables
    full_input = {
        "question": question,
        "context": context
    }
    
    # Run the conversation with the correct input format
    feedback = conversation.run(input=full_input)
    
    # Save the user's response in the memory
    memory.save_context({"user": user_response}, {"bot": feedback})
    return feedback

# Step 6: Example Usage
if __name__ == "__main__":
    print("Welcome to the Job Interview Practice Bot!")
    
    # Example interview loop
    questions = [
        "Can you tell me about yourself?",
        "What is your greatest strength?",
        "Why do you want to work at this company?"
    ]

    for question in questions:
        print(f"\nBot: {question}")
        user_response = input("You: ")
        feedback = interview_bot(question, user_response)
        print(f"Bot Feedback: {feedback}")
