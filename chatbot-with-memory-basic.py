import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

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

# Define a conversation with multiple messages
# NOTE: we can save it into a database and load it later to continue the conversation with the model
conversation = [
    SystemMessage(
        "You are a helpful assistant that can provide information about Python programming language."
    ),
    HumanMessage("What is Python?"),
    AIMessage(
        "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It has a large standard library and a vibrant ecosystem of third-party packages, making it popular for web development, data analysis, artificial intelligence, scientific computing, and more."
    ),
    HumanMessage("When was it released?"),
]

response = llm.invoke(conversation)
print(response.content)
