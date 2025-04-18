import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import numpy as np

from ncf_model import NCFModel
from dataset import NCFDataset
from recommender import get_content_features

# Device setup (M3 Mac doesn't support CUDA)
device = torch.device('cpu')

# Load preprocessed data
content_features = np.load('data/content_matrix.npy')  # Content feature matrix
movie_map = pickle.load(open('data/movie_map.pkl', 'rb'))  # Movie ID to index mapping
num_movies = len(movie_map)

# Load ratings: [userId, movieId, rating]
ratings = np.loadtxt('data/ratings.csv', delimiter=',', skiprows=1)

user_ids = ratings[:, 0].astype(int)
movie_ids = ratings[:, 1].astype(int)

# Adjust movie_ids to match available content features
movie_ids = np.clip(movie_ids, 0, num_movies - 1)

# Get dimensions
embedding_dim = 64
content_dim = content_features.shape[1]
num_users = user_ids.max() + 1
num_items = movie_ids.max() + 1

# Convert to tensors
user_tensor = torch.tensor(user_ids, dtype=torch.long)
item_tensor = torch.tensor(movie_ids, dtype=torch.long)

# Get item-wise content features
all_content = torch.tensor(content_features, dtype=torch.float32)
content_tensor = all_content[item_tensor]

# Create dataset and dataloader
dataset = NCFDataset(user_tensor, item_tensor, content_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
model = NCFModel(num_users, num_items, embedding_dim, content_dim).to(device)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train function
def train_ncf_model():
    epochs = 3  # Reduced from 10 to 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_ids_batch, movie_ids_batch, content_features_batch in dataloader:
            user_ids_batch = user_ids_batch.to(device)
            movie_ids_batch = movie_ids_batch.to(device)
            content_features_batch = content_features_batch.to(device)

            optimizer.zero_grad()
            predictions = model(user_ids_batch, movie_ids_batch, content_features_batch)
            loss = loss_fn(predictions, content_features_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # Save checkpoint
        checkpoint_path = f"data/ncf_model_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)

# Run training
train_ncf_model()