import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI()
