import torch
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    def __init__(self, user_ids, movie_ids, content_features):
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.content_features = content_features  # shape: (num_movies, content_dim)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]

        content_feat = self.content_features[movie_id]  # movie_id is already index-mapped!

        return user_id, movie_id, content_feat