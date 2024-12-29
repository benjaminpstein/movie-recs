import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def recommend_movies(user_input, movie_embeddings, movies_df, model, top_k=20, output_k=5,
                     weight_similarity=0.75, weight_rating=0.25, randomness_factor=0.05):
    user_embedding = model.encode(user_input, convert_to_tensor=True, device=model.device)

    cosine_scores = torch.nn.functional.cosine_similarity(user_embedding, movie_embeddings)


    combined_scores = (weight_similarity * cosine_scores.cpu().numpy() +
                       weight_rating * movies_df['normalized_weighted_rating'].values
                       )

    if np.isnan(combined_scores).sum() > 0:
        raise ValueError("NaN values found in combined scores. Check inputs and normalization.")

    random_noise = np.random.uniform(-randomness_factor, randomness_factor, size=combined_scores.shape)
    randomized_scores = combined_scores + random_noise

    top_indices = randomized_scores.argsort()[-top_k:][::-1]
    top_scores = randomized_scores[top_indices]

    recommendations = movies_df.iloc[top_indices].copy()
    recommendations['score'] = top_scores

    recommendations = recommendations[~recommendations['score'].isna()]

    return recommendations[['title', 'genres', 'score', 'description', 'rich_description']].head(output_k)


def main():
    movies = pd.read_csv('data_processing/rich_processed_movies_with_weighted_ratings.csv')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    movie_embeddings = np.load('embeddings/movie_embeddings.npy')
    movie_embeddings = torch.from_numpy(movie_embeddings).to(device)

    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    while 1:

        user_input = input("Describe the movie you want to watch: ")

        recommendations = recommend_movies(user_input, movie_embeddings, movies, model)

        print("\nRecommended Movies:")
        for idx, row in recommendations.iterrows():
            print(f"- {row['title']} (Score: {row['score']:.4f})")
            print(f"  Genres: {row['genres']}")
            print(f"  Description: {row['description']}\n")

if __name__ == '__main__':
    main()