import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

# Load movie data
movies_df = pd.read_csv('data/movies.csv')
movies_df['title'] = movies_df['title'].str.lower()

# Extract year from title
def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None

movies_df['year'] = movies_df['title'].apply(extract_year)

# Load precomputed content matrix and movie mapping
content_matrix = np.load('data/content_matrix.npy')
movie_map = pickle.load(open('data/movie_map.pkl', 'rb'))

# Load genome scores and tags data
genome_scores_df = pd.read_csv('data/genome-scores.csv')
genome_tags_df = pd.read_csv('data/genome-tags.csv')

# Helper: Get movie index in content matrix
def get_movie_index(title):
    title = title.lower().strip()
    matches = movies_df[movies_df['title'] == title]
    if matches.empty:
        return None
    movie_id = matches.iloc[0]['movieId']
    return movie_map.get(movie_id)

# Hybrid recommender using content matrix
def hybrid_recommender(title, top_n=10):
    idx = get_movie_index(title)
    if idx is None:
        return []
    similarities = cosine_similarity([content_matrix[idx]], content_matrix)[0]
    similar_indices = similarities.argsort()[::-1][1:top_n + 1]
    return [movies_df.iloc[i]['title'].title() for i in similar_indices]

# Genre-based recommender
def genre_based_recommender(title, top_n=10):
    title = title.lower().strip()
    if title not in movies_df['title'].values:
        return []
    target_genres = set(movies_df[movies_df['title'] == title]['genres'].values[0].split('|'))
    df = movies_df.copy()
    df['genre_overlap'] = df['genres'].apply(lambda x: len(target_genres & set(x.split('|'))))
    recommendations = df[df['title'] != title].sort_values(by='genre_overlap', ascending=False)
    return recommendations['title'].head(top_n).str.title().tolist()

# Year-based recommender
def year_based_recommender(title, top_n=10):
    title = title.lower().strip()
    matches = movies_df[movies_df['title'] == title]
    if matches.empty:
        return []

    movie_year = matches.iloc[0]['year']
    if pd.isna(movie_year):
        return []

    recommendations = movies_df[(movies_df['year'] == movie_year) & (movies_df['title'] != title)]
    return recommendations['title'].head(top_n).str.title().tolist()

# Genome-tag-based recommender with top tags included
def genome_tag_based_recommender(title, top_n=10, top_tags=5):
    title = title.lower().strip()
    matches = movies_df[movies_df['title'] == title]
    if matches.empty:
        return {'top_tags': [], 'recommendations': []}

    movie_id = matches.iloc[0]['movieId']
    if movie_id not in genome_scores_df['movieId'].values:
        return {'top_tags': [], 'recommendations': []}

    # Pivot to movieId x tagId matrix
    tag_matrix = genome_scores_df.pivot_table(index='movieId', columns='tagId', values='relevance', fill_value=0)
    if movie_id not in tag_matrix.index:
        return {'top_tags': [], 'recommendations': []}

    # Find top genome tags by relevance for the input movie
    movie_vector = tag_matrix.loc[movie_id]
    top_tag_ids = movie_vector.sort_values(ascending=False).head(top_tags).index.tolist()
    top_tags = genome_tags_df[genome_tags_df['tagId'].isin(top_tag_ids)]['tag'].tolist()

    # Compute cosine similarity to find similar movies
    movie_vector = movie_vector.values.reshape(1, -1)
    similarities = cosine_similarity(movie_vector, tag_matrix)[0]

    similar_movie_ids = tag_matrix.index[np.argsort(similarities)[::-1]]
    similar_movie_ids = [mid for mid in similar_movie_ids if mid != movie_id][:top_n]

    recommended_titles = movies_df[movies_df['movieId'].isin(similar_movie_ids)]['title'].dropna().str.title().tolist()

    return {'top_tags': top_tags, 'recommendations': recommended_titles}