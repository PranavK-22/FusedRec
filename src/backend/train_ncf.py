import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import numpy as np

from ncf_model import NCFModel
from dataset import NCFDataset
from recommender import get_content_features

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load preprocessed data
content_features = np.load('data/content_matrix.npy')
movie_map = pickle.load(open('data/movie_map.pkl', 'rb'))
num_movies = len(movie_map)
ratings = np.loadtxt('data/ratings.csv', delimiter=',', skiprows=1)

user_ids = ratings[:, 0].astype(int)
movie_ids = ratings[:, 1].astype(int)
ratings_vals = ratings[:, 2].astype(np.float32)

movie_ids = np.clip(movie_ids, 0, num_movies - 1)

embedding_dim = 64
content_dim = content_features.shape[1]
num_users = user_ids.max() + 1
num_items = movie_ids.max() + 1

user_tensor = torch.tensor(user_ids, dtype=torch.long)
item_tensor = torch.tensor(movie_ids, dtype=torch.long)
ratings_tensor = torch.tensor(ratings_vals, dtype=torch.float32)

all_content = torch.tensor(content_features, dtype=torch.float32)
content_tensor = all_content[item_tensor]

# Create dataset and dataloader with larger batch size
dataset = NCFDataset(user_tensor, item_tensor, content_tensor, ratings_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

# Initialize model
model = NCFModel(num_users, num_items, embedding_dim, content_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def train_ncf_model():
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_batch, movie_batch, content_batch, ratings_batch in dataloader:
            user_batch = user_batch.to(device)
            movie_batch = movie_batch.to(device)
            content_batch = content_batch.to(device)
            ratings_batch = ratings_batch.to(device)

            optimizer.zero_grad()
            predictions = model(user_batch, movie_batch, content_batch).squeeze()
            loss = loss_fn(predictions, ratings_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        torch.save(model.state_dict(), f"data/ncf_model_epoch_{epoch + 1}.pt")

train_ncf_model()
