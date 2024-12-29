from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=open_ai_key)

batch_input_file_id = 'file-1fkfXqCKDhVn817AjyiVxd'
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)