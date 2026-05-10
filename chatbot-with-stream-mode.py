
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()

# read env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL_IDENTIFIER = os.getenv("OPENAI_MODEL_IDENTIFIER", "")


llm = ChatOpenAI(
    model=OPENAI_MODEL_IDENTIFIER,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,  # pyright: ignore
    temperature=0.1,
)

for chunk in llm.stream('Hello, what is Python?'):
    print(chunk.content, end='', flush=True)
