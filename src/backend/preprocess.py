import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

# Load the movies dataset
movies_df = pd.read_csv('data/movies.csv')

# 1. Process genres to one-hot encode them using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(movies_df['genres'].str.split('|'))
print(f"Genres encoded: {genres_encoded.shape}")

# 2. Create the content matrix (in this case, just genres for now)
# You can add additional features (like genome scores or other movie features) here
content_matrix = genres_encoded  # Using genre features for content-based filtering

# Save the content matrix
np.save('data/content_matrix.npy', content_matrix)

# 3. Create a mapping from movieId to index in the content matrix
movie_map = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}

# Save the movie map
with open('data/movie_map.pkl', 'wb') as f:
    pickle.dump(movie_map, f)

# 4. Create a mapping from movieId to movie title
movieId_title_map = {movie_id: title for movie_id, title in zip(movies_df['movieId'], movies_df['title'])}

# Save the movieId to title map
with open('data/movieId_title_map.pkl', 'wb') as f:
    pickle.dump(movieId_title_map, f)

print("Preprocessing complete. Files saved to 'data/'")