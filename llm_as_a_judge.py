import openai
from dotenv import load_dotenv
import os
import re

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

Also keep in mind if they have actually tried to solve the case or just given up.
If you detect that no proper attemt for the case was done, score 0/10.

To assist you in grading the interview, you are given three example interviews. You are to understand their grading, and provide a grade for the fourth interview with feedback structured similarly.

Example Interview 1:
[
  {"role": "assistant", "content": "How would you approach the problem of declining sales for a retail client?"},
  {"role": "user", "content": "Umm… well, maybe they should just lower prices? That usually works, right?"},
  {"role": "assistant", "content": "Could you elaborate on why that might work or any other factors to consider?"},
  {"role": "user", "content": "I don't know. I just think if stuff is cheaper, more people will buy it."},
  {"role": "assistant", "content": "What if price isn't the main issue?"},
  {"role": "user", "content": "I'm not sure. I guess they could figure out something else, like ads?"}
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
  {"role": "assistant", "content": "How would you approach the problem of declining sales for a retail client?"},
  {"role": "user", "content": "I would start by looking at their pricing strategy, customer demographics, and marketing efforts."},
  {"role": "assistant", "content": "Could you explain how you'd analyze those areas?"},
  {"role": "user", "content": "Sure. For pricing, I'd compare with competitors. For demographics, I'd see if they're targeting the right audience. For marketing, I'd review their campaigns."},
  {"role": "assistant", "content": "What would be your recommendation if pricing seems competitive, but sales are still falling?"},
  {"role": "user", "content": "Um… I'm not sure, but maybe they could try to advertise more?"},
  {"role": "assistant", "content": "And what if advertising doesn't work?"},
  {"role": "user", "content": "They could explore other options, but I'd need more details to say."}
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
  {"role": "assistant", "content": "How would you approach the problem of declining sales for a retail client?"},
  {"role": "user", "content": "I'd start by breaking the problem into three key areas: internal operations, external market factors, and customer insights. Each area provides a potential explanation for declining sales."},
  {"role": "assistant", "content": "That's a good structure. Could you walk me through how you'd analyze those areas?"},
  {"role": "user", "content": "For internal operations, I'd review pricing strategies, product quality, and the customer experience both in-store and online. For external factors, I'd analyze market trends, competitor positioning, and macroeconomic influences. For customer insights, I'd focus on feedback, purchasing behavior, and demographic shifts to understand their evolving needs."},
  {"role": "assistant", "content": "What would you recommend if competition is offering lower prices and better service?"},
  {"role": "user", "content": "I'd recommend differentiating through added value rather than competing on price. For example, launching a loyalty program, enhancing the shopping experience, or targeting niche segments with tailored products."},
  {"role": "assistant", "content": "And if customer feedback suggests dissatisfaction with product quality?"},
  {"role": "user", "content": "I'd prioritize improving quality, even if it means a slight increase in costs, as long-term trust and loyalty outweigh short-term margins."}
]

Feedback
Structure: The interviewee provided a comprehensive and logical framework.
Communication: Ideas were clearly articulated with relevant detail.
Problem-solving: Demonstrated strong analytical skills and proposed actionable, thoughtful recommendations.
Creativity: Suggested innovative strategies like personalized shopping experiences.
Overall Impression: The candidate was highly impressive and demonstrated the qualities of a top-tier consultant.

Rating: 9/10

When grading the interviewee keep in mind that their text input is done through a Speech-to-text module that doesnt always work perfectly.
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
  
def judge_single_answer(answer):
    """
    Uses a simpler prompt to get a rating from 1 to 10 for a single user response.
    Returns (score, explanation_text).
    You can modify the prompt or logic as you wish.
    """
    prompt = f"""You are a concise judge. 
    You will receive a single user answer (one short utterance).
    You must rate it on a scale of 1-10, then provide a one-line explanation.

    User's answer:
    \"{answer}\"

    Output format example (only this format):
    Rating: 7
    Explanation: Clear structure but needs more examples.
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    response_text = completion.choices[0].message.content

    # Attempt to parse rating from the response
    # e.g. "Rating: 7\nExplanation: Clear structure but needs more examples."
    rating_match = re.search(r"Rating:\s*(\d+)", response_text)
    explanation = ""
    if rating_match:
        rating = int(rating_match.group(1))
    else:
        rating = 5  # fallback if we can't parse
    # parse explanation
    expl_match = re.search(r"Explanation:\s*(.*)", response_text)
    if expl_match:
        explanation = expl_match.group(1).strip()

    return rating, explanation


if __name__ == "__main__":
    interview = "Interviewer: Hello, thank you for joining us today. Can you please introduce yourself?"
    print(insert_into_prompt_template(interview))