import pandas as pd
import json

jsonl_file = 'batch_output.jsonl'
jsonl_data = []

with open(jsonl_file, 'r') as file:
    for line in file:
        jsonl_data.append(json.loads(line))

jsonl_extracted = []
for entry in jsonl_data:
    custom_id = entry.get('custom_id')
    response_body = entry.get('response', {}).get('body', {})
    choices = response_body.get('choices', [])
    if choices:
        rich_description = choices[0].get('message', {}).get('content', '')
        jsonl_extracted.append({'imdbId': int(custom_id), 'rich_description': rich_description})

jsonl_df = pd.DataFrame(jsonl_extracted)

csv_file = '../data_processing/processed_movies.csv'
csv_df = pd.read_csv(csv_file)

merged_df = pd.merge(csv_df, jsonl_df, on='imdbId', how='left')

output_file = '../data_processing/rich_processed_movies.csv'
merged_df.to_csv(output_file, index=False)
