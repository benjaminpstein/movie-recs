from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=open_ai_key)

output_file_id = 'file-LnLd4w9Wch5WjHi2o8NZnh'

file_content = client.files.content(output_file_id)

local_filename = 'batch_output.jsonl'

with open(local_filename, 'wb') as local_file:
    local_file.write(file_content.read())