import torch
import pickle
import numpy as np
import pandas as pd

from ncf_model import NCFModel

def load_model_and_data(checkpoint_path="data/ncf_model_epoch_55.pt"):
    # Load supporting data
    movie_map = pickle.load(open("data/movie_map.pkl", "rb"))
    movieId_title_map = pickle.load(open("data/movieId_title_map.pkl", "rb"))
    content_matrix = np.load("data/content_matrix.npy")
    movies_df = pd.read_csv("data/movies.csv")

    # Set dimensions
    embedding_dim = 64
    content_dim = content_matrix.shape[1]
    num_users = 138494  # or load dynamically if stored
    num_items = len(movie_map)

    # Initialize model and load weights
    model = NCFModel(num_users, num_items, embedding_dim, content_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    model.eval()

    return model, movie_map, movieId_title_map, content_matrix, movies_df