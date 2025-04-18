import numpy as np
import torch
import pickle
import pandas as pd

from model_loader import load_model_and_data

class HybridRecommender:
    def __init__(self):
        self.model, self.movie_map, self.movieId_title_map, self.content_matrix, self.movies_df = load_model_and_data()
        self.model.eval()

    def recommend(self, movie_title, top_k=10):
        # Map lowercase title to movieId
        reverse_map = {v.lower(): k for k, v in self.movieId_title_map.items()}
        movie_id = reverse_map.get(movie_title.lower())
        if movie_id is None:
            return []

        # Get index of the movie in the content matrix
        movie_idx = self.movie_map.get(movie_id)
        if movie_idx is None:
            return []

        # Compute similarity using dot product of content features
        query_vector = self.content_matrix[movie_idx]
        similarities = np.dot(self.content_matrix, query_vector)

        # Get top similar movie indices (excluding the input movie itself)
        similar_indices = similarities.argsort()[::-1][1:top_k + 1]

        similar_titles = []
        for idx in similar_indices:
            mid = list(self.movie_map.keys())[list(self.movie_map.values()).index(idx)]
            row = self.movies_df[self.movies_df['movieId'] == mid]
            if not row.empty:
                title = row.iloc[0]['title']
                similar_titles.append(title)

        return similar_titles


# Optional: expose the content feature extraction function if needed elsewhere
def get_content_features(movie_ids, content_matrix, movie_map):
    indices = [movie_map.get(mid, -1) for mid in movie_ids]
    indices = [idx for idx in indices if idx != -1]
    return content_matrix[indices]