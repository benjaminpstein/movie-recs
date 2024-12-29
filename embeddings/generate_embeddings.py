import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    movies = pd.read_csv('../data_processing/rich_processed_movies.csv')

    required_columns = ['genres', 'rich_description']
    for col in required_columns:
        if col not in movies.columns:
            raise ValueError(f"Missing required column: {col}")

    movies[required_columns] = movies[required_columns].fillna('')

    movies['combined_text'] = movies.apply(
        lambda row: ' '.join([row['genres'], row['rich_description']]),
        axis=1
    )

    model = SentenceTransformer('all-MiniLM-L6-v2')

    combined_texts = movies['combined_text'].tolist()
    print("Generating embeddings for combined movie information...")
    movie_embeddings = model.encode(combined_texts, convert_to_tensor=False)

    np.save('movie_embeddings.npy', movie_embeddings)
    print("Embeddings generated and saved to 'movie_embeddings.npy'.")

if __name__ == '__main__':
    main()