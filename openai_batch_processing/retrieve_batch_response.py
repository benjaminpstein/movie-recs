from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=open_ai_key)

file_response = client.files.content("")

print(file_response.text)