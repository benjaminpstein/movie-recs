from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=open_ai_key)

batch_input_file = client.files.create(
    file=open("movies_for_chat.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)