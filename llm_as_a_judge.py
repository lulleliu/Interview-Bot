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
[
  {"role": "user", "content": "How would you approach the problem of declining sales for a retail client?"},
  {"role": "assistant", "content": "Umm… well, maybe they should just lower prices? That usually works, right?"},
  {"role": "user", "content": "Could you elaborate on why that might work or any other factors to consider?"},
  {"role": "assistant", "content": "I don't know. I just think if stuff is cheaper, more people will buy it."},
  {"role": "user", "content": "What if price isn't the main issue?"},
  {"role": "assistant", "content": "I'm not sure. I guess they could figure out something else, like ads?"}
]

Feedback
Structure: The interviewee did not attempt to structure their approach to the problem.
Communication: The responses were vague and unorganized.
Problem-solving: Limited effort in addressing the problem, with no consideration of other factors or alternatives.
Creativity: No evidence of creative or critical thinking.
Overall Impression: The candidate lacked focus and did not demonstrate basic problem-solving or communication skills.

Rating: 2/10


Example Interview 2:
[
  {"role": "user", "content": "How would you approach the problem of declining sales for a retail client?"},
  {"role": "assistant", "content": "I would start by looking at their pricing strategy, customer demographics, and marketing efforts."},
  {"role": "user", "content": "Could you explain how you'd analyze those areas?"},
  {"role": "assistant", "content": "Sure. For pricing, I'd compare with competitors. For demographics, I'd see if they're targeting the right audience. For marketing, I'd review their campaigns."},
  {"role": "user", "content": "What would be your recommendation if pricing seems competitive, but sales are still falling?"},
  {"role": "assistant", "content": "Um… I'm not sure, but maybe they could try to advertise more?"},
  {"role": "user", "content": "And what if advertising doesn't work?"},
  {"role": "assistant", "content": "They could explore other options, but I'd need more details to say."}
]

Feedback
Structure: The interviewee provided a somewhat structured approach but lacked depth.
Communication: Ideas were conveyed clearly but lacked detail.
Problem-solving: Showed basic analytical thinking but did not fully address the problem or provide actionable recommendations.
Creativity: Suggestions were generic and lacked innovative thinking.
Overall Impression: The candidate demonstrated basic competency but did not stand out.

Rating: 5/10

Example Interview 3:
[
  {"role": "user", "content": "How would you approach the problem of declining sales for a retail client?"},
  {"role": "assistant", "content": "I'd start by breaking the problem into three key areas: internal operations, external market factors, and customer insights. Each area provides a potential explanation for declining sales."},
  {"role": "user", "content": "That's a good structure. Could you walk me through how you'd analyze those areas?"},
  {"role": "assistant", "content": "For internal operations, I'd review pricing strategies, product quality, and the customer experience both in-store and online. For external factors, I'd analyze market trends, competitor positioning, and macroeconomic influences. For customer insights, I'd focus on feedback, purchasing behavior, and demographic shifts to understand their evolving needs."},
  {"role": "user", "content": "What would you recommend if competition is offering lower prices and better service?"},
  {"role": "assistant", "content": "I'd recommend differentiating through added value rather than competing on price. For example, launching a loyalty program, enhancing the shopping experience, or targeting niche segments with tailored products."},
  {"role": "user", "content": "And if customer feedback suggests dissatisfaction with product quality?"},
  {"role": "assistant", "content": "I'd prioritize improving quality, even if it means a slight increase in costs, as long-term trust and loyalty outweigh short-term margins."}
]

Feedback
Structure: The interviewee provided a comprehensive and logical framework.
Communication: Ideas were clearly articulated with relevant detail.
Problem-solving: Demonstrated strong analytical skills and proposed actionable, thoughtful recommendations.
Creativity: Suggested innovative strategies like personalized shopping experiences.
Overall Impression: The candidate was highly impressive and demonstrated the qualities of a top-tier consultant.

Rating: 9/10

This is the conversation between the interviewer and the interviewee you are grading:

-----
{REPLACE__WITH__INTERVIEW}
-----
"""
    return prompt_template.replace("{REPLACE__WITH__INTERVIEW}", interview)

def get_judge_response(interview):
    prompt = insert_into_prompt_template(interview)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Use the GPT-4o-mini model
        messages= [{"role": "system", "content": prompt}]
        # to add temperature setting?
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    interview = "Interviewer: Hello, thank you for joining us today. Can you please introduce yourself?"
    print(insert_into_prompt_template(interview))