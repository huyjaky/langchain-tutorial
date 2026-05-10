from dataclasses import dataclass
import requests
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.tools import ToolRuntime

from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

# read env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL_IDENTIFIER = os.getenv("OPENAI_MODEL_IDENTIFIER", "") 


@tool("get_weather", description="Get the current weather for a given location.")
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()


llm = ChatOpenAI(
    model = OPENAI_MODEL_IDENTIFIER,
    base_url = OPENAI_BASE_URL,
    api_key = OPENAI_API_KEY, # pyright: ignore
    temperature=0.3,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm, 
    tools=[get_weather],
    system_prompt="You are a helpful assistant that can provide weather information for any city. Use the get_weather tool to fetch the current weather data when asked.",
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the current weather in Da Nang, Viet Nam today"}
        ],
    }
)

# print(response)
print(response['messages'][-1].content)
