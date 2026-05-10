
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from langchain_openai import ChatOpenAI

load_dotenv()

# read env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL_IDENTIFIER = os.getenv("OPENAI_MODEL_IDENTIFIER", "") 


llm = ChatOpenAI(
    model = OPENAI_MODEL_IDENTIFIER,
    base_url = OPENAI_BASE_URL,
    api_key = OPENAI_API_KEY, # pyright: ignore
    temperature=0.1,
)

response = llm.invoke('hello whats python?')
print(response.content)
