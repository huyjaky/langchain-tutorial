from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from dotenv import load_dotenv
import os
from rich import print

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector

load_dotenv()


# read env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL_IDENTIFIER = os.getenv("OPENAI_MODEL_IDENTIFIER", "")
CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")  # Or whatever you named it

embeddings = OpenAIEmbeddings(
    model="text-embedding-bge-m3",
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,  # pyright: ignore
    check_embedding_ctx_length=False,
)

texts = [
    "The cat is on the table.",
    "The dog is in the garden.",
    "The bird is flying in the sky.",
    "The fish is swimming in the pond.",
]

vector_store = PGVector.from_texts(
    texts,
    embeddings,
    collection_name="test_collection",
    connection_string=CONNECTION_STRING,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    name="vector_store_retriever",
    description="Use this tool to retrieve relevant information from the vector store based on a query.",
)

llm = ChatOpenAI(
    model=OPENAI_MODEL_IDENTIFIER,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,  # pyright: ignore
)

agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt="You are a helpful assistant that can retrieve relevant information from a vector store. Use the vector_store_retriever tool to fetch relevant information when asked.",
)

results = agent.invoke({"messages": [{"role": "user", "content": "Where is the cat?"}]})
print(results["messages"][-1].content)
