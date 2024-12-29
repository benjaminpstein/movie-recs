from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=open_ai_key)

batches = client.batches.list()

for batch in batches:
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created At: {batch.created_at}")
