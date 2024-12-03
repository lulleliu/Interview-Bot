import os
from datetime import datetime
import openai
from IPython.display import HTML, display
from ipywidgets import widgets
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

loading_bar = None
chatbot = None
chatbot_agent_name = None
chatbot_user_name = None
current_prompt_display = None


def display_human_message(message):
    m = (
        f'<div class="chat-message-right pb-4"><div>'
        + f'<img src="images/bear.png" class="rounded-circle mr-1" width="40" height="40">'
        + f'<div class="text-muted small text-nowrap mt-2">{datetime.now().strftime("%H:%M:%S")}</div></div>'
        + '<div class="flex-shrink-1 bg-light rounded py-2 px-3 ml-3">'
        + f'<div class="font-weight-bold mb-1">{chatbot_user_name}</div>{message.content}</div>'
    )
    output.append_display_data(HTML(m))

def display_bot_message(message):
    answer_formatted = message.content.replace("$", r"\$")
    if hasattr(message, 'tool_calls') and isinstance(message.tool_calls, list) and message.tool_calls:
        answer_formatted = "<i>Tool calls</i>: "
        for tool_call in message.tool_calls:
            answer_formatted += f"{tool_call} "
    m = (
        f'<div class="chat-message-left pb-4"><div>'
        + f'<img src="images/cat.png" class="rounded-circle mr-1" width="40" height="40">'
        + f'<div class="text-muted small text-nowrap mt-2">{datetime.now().strftime("%H:%M:%S")}</div></div>'
        + '<div class="flex-shrink-1 bg-light rounded py-2 px-3 ml-3">'
        + f'<div class="font-weight-bold mb-1">{chatbot_agent_name}</div>{answer_formatted}</div>'
    )
    output.append_display_data(HTML(m))


def display_tool_message(message):
    answer_formatted = message.content.replace("$", r"\$")
    m = (
        f'<div class="chat-message-left pb-4"><div>'
        + f'<img src="images/cat.png" class="rounded-circle mr-1" width="40" height="40">'
        + f'<div class="text-muted small text-nowrap mt-2">{datetime.now().strftime("%H:%M:%S")}</div></div>'
        + '<div class="flex-shrink-1 bg-light rounded py-2 px-3 ml-3">'
        + f'<div class="font-weight-bold mb-1">Tool</div>{answer_formatted}</div>'
    )
    output.append_display_data(HTML(m))    

def display_system_message(message):
    prompt = message.content.replace("\n", "<br>")
    m = (
        f'<div class="chat-message-wide pb-4"><div class="flex-shrink-1 bg-light rounded py-2 px-3 ml-3" style="width:100%"><div class="font-weight-bold mb-1">System prompt</div>{prompt}</div></div>'
    )
    output.append_display_data(HTML(m))    



def update_history(history):
    output.outputs = []
    for message in history:
        if isinstance(message, HumanMessage):
            display_human_message(message)
        elif isinstance(message, SystemMessage):
            display_system_message(message)
        elif isinstance(message, ToolMessage):
            display_tool_message(message)
        elif isinstance(message, AIMessage):
            display_bot_message(message)


def text_eventhandler(*args):
    # Needed bc when we "reset" the text input
    # it fires instantly another event since
    # we "changed" it's value to ""
    if args[0]["new"] == "":
        return

    # Show loading animation
    loading_bar.layout.display = "block"

    # Get question
    question = args[0]["new"]

    chatbot.question(question)

    # Reset text field
    args[0]["owner"].value = ""

    update_history(chatbot.prompt())

    try:
        chatbot.answer(question)
    except Exception as e:
        chatbot.history.append(AIMessage(content="Error: " + str(e)))

    update_history(chatbot.prompt())    

    # Turn off loading animation
    loading_bar.layout.display = "none"



def restart_chat(b):
    chatbot.reset()
    output.outputs = []
    current_prompt_display.value = ""  # Clear the current prompt display


def start_chat(bot, agent_name="LLM", user_name="You"):
    global chatbot, output, chatbot_agent_name, chatbot_user_name, current_prompt_display
    chatbot = bot
    chatbot_agent_name = agent_name
    chatbot_user_name = user_name

    output = widgets.Output()
    update_history(chatbot.prompt())

    in_text = widgets.Text()
    in_text.continuous_update = False
    in_text.observe(text_eventhandler, "value")

    current_prompt_display = widgets.Textarea(
        value="",
        placeholder="",
        disabled=True,
        layout=widgets.Layout(width="100%", height="100px")
    )

    global loading_bar
    loading_bar = widgets.Image(
        value=open("images/loading.gif", "rb").read(), 
        format="gif", width="20", height="20", layout={"display": "None"}
    )

    restart_button = widgets.Button(description="Restart")
    restart_button.on_click(restart_chat)

    display(HTML(open('jupyter_chat.html', 'r').read()))
    #display(widgets.VBox([widgets.Label(value="System message:"), current_prompt_display]))
    display(
        widgets.HBox(
            [output],
            layout=widgets.Layout(
                width="100%",
                max_height="500px",
                display="inline-flex",
                flex_flow="column-reverse",
            ),
        )
    )
    display(
        widgets.Box(
            children=[loading_bar, in_text, restart_button],
            layout=widgets.Layout(display="flex", flex_flow="row"),
        )
    )