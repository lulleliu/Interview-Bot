import openai
from dotenv import load_dotenv
import os

load_dotenv()

client = openai.OpenAI()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

def insert_into_prompt_template(interview=""):
    prompt_template = """You are a judge with the purpose of grading a case interview between 1 and 10.
The interviewee is a candidate for a consulting position at a top firm.

To grade the interview, you should consider the following criteria:
- Structure: Did the interviewee structure the problem effectively?
- Communication: Did the interviewee communicate their thoughts clearly?
- Problem-solving: Did the interviewee solve the problem effectively?
- Creativity: Did the interviewee demonstrate creative thinking?
- Overall impression: What is your overall impression of the interviewee's performance?

To assist you in grading the interview, you are given three example interviews. You are to understand their grading, and provide a grade for the fourth interview with feedback structured similarly.

Example Interview 1:
2/10

Example Interview 2:
5/10

Example Interview 3:
9/10

This is the conversation between the interviewer and the interviewee you are grading:

-----
{0}
-----
"""
    return prompt_template.format(interview)

def get_judge_response(interview):
    prompt = insert_into_prompt_template(interview)
    response = client.chat(prompt)
    return response

if __name__ == "__main__":
    interview = "Interviewer: Hello, thank you for joining us today. Can you please introduce yourself?"
    print(insert_into_prompt_template(interview))