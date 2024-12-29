import pandas as pd
import requests
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os

load_dotenv()
tmdb_key = os.getenv("TMDB_KEY")

def fetch_description(tmdb_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    params = {
        'api_key': api_key,
        'language': 'en-US'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('overview', '')
    else:
        return ''


def main():
    # Load movies and tags data
    movies = pd.read_csv('../ml-latest-small/movies.csv')
    tags = pd.read_csv('../ml-latest-small/tags.csv')
    links = pd.read_csv('../ml-latest-small/links.csv')
    ratings = pd.read_csv('../ml-latest-small/ratings.csv')

    # Merge movies with tags
    movies_with_tags = pd.merge(movies, tags, on='movieId', how='left')

    # Aggregate tags for each movie
    movie_tags = movies_with_tags.groupby('movieId')['tag'].apply(
        lambda x: ' '.join(x.dropna().unique())
    ).reset_index()

    # Merge aggregated tags back into movies DataFrame
    movies = pd.merge(movies, movie_tags, on='movieId', how='left')

    # Fill NaN values in 'tag' column with empty strings
    movies['tag'] = movies['tag'].fillna('')

    # Merge with links to get tmdbId/imdbId
    movies = pd.merge(movies, links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')

    # Fill NaN tmdbId/imdbId with 0 and convert to integer
    movies['tmdbId'] = movies['tmdbId'].fillna(0).astype(int)
    movies['imdbId'] = movies['imdbId'].fillna(0).astype(int)

    # Remove movies without tmdbId/imdbId
    movies = movies[movies['tmdbId'] != 0]
    movies = movies[movies['imdbId'] != 0]

    # Fetch movie descriptions from TMDb
    print("Fetching movie descriptions from TMDb...")
    descriptions = []
    for tmdb_id in tqdm(movies['tmdbId']):
        description = fetch_description(tmdb_id, tmdb_key)
        descriptions.append(description)
        time.sleep(0.1)

    movies['description'] = descriptions

    # Combine tags and descriptions
    movies['description'] = movies.apply(
        lambda x: f"{x['tag']} {x['description']}".strip(),
        axis=1
    )

    # For movies without descriptions, use title and genres
    movies['description'] = movies.apply(
        lambda x: x['description'] if x['description'].strip() else f"{x['title']} {x['genres']}",
        axis=1
    )

    # Calculate average rating and number of ratings for each movie
    ratings_grouped = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    # Merge the average rating and number of ratings into the movies DataFrame
    movies = pd.merge(movies, ratings_grouped, on='movieId', how='left')

    # Save the processed data to a CSV file
    movies.to_csv('processed_movies.csv', index=False)


    print("Data preparation complete. Processed data saved to 'processed_movies.csv'.")


if __name__ == '__main__':
    main()