import csv
import json

csv_file_path = '../data_processing/processed_movies.csv'
jsonl_file_path = 'movies_for_chat.jsonl'

with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
        for row in csv_reader:
            json_object = {
                'custom_id': f"{row['imdbId']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body":{
                    'model': 'gpt-4o-mini',
                    'messages': [
                        {'role': 'system', 'content': 'You are an assistant that generates vibe-based movie descriptions.'},
                        {'role': 'user',
                         'content': f"Given the following details, describe the vibe of this movie in 2-3 sentences. Include descriptions of what the movie looks like, how it makes you feel, and who (if anyone) to watch it with: {row['title']} {row['genres']} {row['tag']} {row['description']}"}
                    ]
                }
            }
            jsonl_file.write(json.dumps(json_object) + '\n')
