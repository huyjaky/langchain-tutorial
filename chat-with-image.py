from base64 import b64encode
from dotenv import load_dotenv
import os

load_dotenv()
import base64
import requests


from langchain_openai import ChatOpenAI

# read env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL_IDENTIFIER = os.getenv("OPENAI_MODEL_IDENTIFIER", "")

llm = ChatOpenAI(
    model=OPENAI_MODEL_IDENTIFIER,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,  # pyright: ignore
    temperature=0.3,
)


def create_image_message(image_url: str):
    response = requests.get(image_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "image/png")
    base64_encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:{content_type};base64,{base64_encoded}"


# NOTE: we have two ways to send image to the model, one is by sending the image url, 
# the other is by sending the base64 encoded image content. Both of them work fine, 
# you can choose one of them or try both of them to see the difference in response time 
# and response quality.

message_w_link = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": create_image_message(
                    "https://cdn2.fptshop.com.vn/unsafe/Uploads/images/tin-tuc/164679/Originals/facebook-la-gi-1.jpg"
                ),
            },
        },
    ],
}

message_w_local_image = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image",
            "base64": b64encode(
                open("./attachs/0897b7bb809c0fc2568d.jpg", "rb").read()
            ).decode(),
            "mime_type": "image/jpg",
        },
    ],
}

response = llm.invoke([message_w_link])
print(response.content)
