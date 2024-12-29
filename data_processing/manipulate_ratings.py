import pandas as pd

data = pd.read_csv("rich_processed_movies.csv")

required_columns = ['avg_rating', 'num_ratings']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

data['avg_rating'] = data['avg_rating'].fillna(0)
data['num_ratings'] = data['num_ratings'].fillna(0)

C = data['avg_rating'].mean()

m = data['num_ratings'].quantile(0.5)

data['weighted_rating'] = (
    (data['avg_rating'] * data['num_ratings'] + C * m) /
    (data['num_ratings'] + m)
)

min_rating = data['weighted_rating'].min()
max_rating = data['weighted_rating'].max()

data['normalized_weighted_rating'] = (
    (data['weighted_rating'] - min_rating) /
    (max_rating - min_rating)
)

output_file = 'rich_processed_movies_with_weighted_ratings.csv'
data.to_csv(output_file, index=False)