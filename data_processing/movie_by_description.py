import pandas as pd

# Load movies and tags data
movies = pd.read_csv('../ml-latest-small/movies.csv')
tags = pd.read_csv('../ml-latest-small/tags.csv')

movies_with_tags = pd.merge(movies, tags, on='movieId', how='left')

movie_tags = movies_with_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().unique())).reset_index()

movies = pd.merge(movies, movie_tags, on='movieId', how='left')

movies['tag'] = movies['tag'].fillna('')

movies['description'] = movies['tag']

movies['description'] = movies.apply(
    lambda x: x['description'] if x['description'] else f"{x['title']} {x['genres']}", axis=1)

from sentence_transformers import SentenceTransformer
import torch

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the movie descriptions
movie_descriptions = movies['description'].tolist()
movie_embeddings = model.encode(movie_descriptions, convert_to_tensor=True)


def recommend_movies(user_input, movie_embeddings, movies_df, top_k=5):
    # Encode the user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = torch.nn.functional.cosine_similarity(user_embedding, movie_embeddings)

    # Get the top_k movies
    top_results = torch.topk(cosine_scores, k=top_k)
    top_indices = top_results.indices.cpu().numpy()
    scores = top_results.values.cpu().numpy()

    # Retrieve and display the recommended movies
    recommendations = movies_df.iloc[top_indices]
    recommendations = recommendations.copy()
    recommendations['score'] = scores

    return recommendations[['title', 'genres', 'score']]

user_input = input("Describe the movie you want to watch: ")

# Get recommendations
recommendations = recommend_movies(user_input, movie_embeddings, movies)

# Display the recommendations
print("\nRecommended Movies:")
for idx, row in recommendations.iterrows():
    print(f"- {row['title']} (Score: {row['score']:.4f})\n  Genres: {row['genres']}")