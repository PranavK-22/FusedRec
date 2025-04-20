import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import pickle
import numpy as np
import os
from tqdm import tqdm

from ncf_model import NCFModel
from dataset import NCFDataset

# Select GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Control training range


checkpoint_epoch = 45  # Starting from after this epoch
end_epoch = 55

# Load paths
checkpoint_path = f"data/ncf_model_epoch_{checkpoint_epoch}.pt"
content_features = np.load('data/content_matrix.npy')
movie_map = pickle.load(open('data/movie_map.pkl', 'rb'))
ratings = np.loadtxt('data/ratings.csv', delimiter=',', skiprows=1)

# Parse and preprocess data
user_ids = ratings[:, 0].astype(int)
movie_ids = ratings[:, 1].astype(int)
ratings_values = ratings[:, 2].astype(np.float32)

num_movies = len(movie_map)
movie_ids = np.clip(movie_ids, 0, num_movies - 1)

embedding_dim = 64
content_dim = content_features.shape[1]
num_users = user_ids.max() + 1
num_items = movie_ids.max() + 1

user_tensor = torch.tensor(user_ids, dtype=torch.long)
item_tensor = torch.tensor(movie_ids, dtype=torch.long)
ratings_tensor = torch.tensor(ratings_values, dtype=torch.float32).view(-1, 1)
all_content = torch.tensor(content_features, dtype=torch.float32)
content_tensor = all_content[item_tensor]

# Dataset and DataLoaders
full_dataset = NCFDataset(user_tensor, item_tensor, content_tensor, ratings_tensor)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Model setup
model = NCFModel(num_users, num_items, embedding_dim, content_dim).to(device)
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from epoch {checkpoint_epoch}")
else:
    print(f"Checkpoint not found: {checkpoint_path}")
    exit()

optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

# Training loop with validation and tqdm
def continue_training(start_epoch, end_epoch):
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for user_batch, item_batch, content_batch, rating_batch in train_bar:
            user_batch = user_batch.to(device)
            item_batch = item_batch.to(device)
            content_batch = content_batch.to(device)
            rating_batch = rating_batch.to(device)

            optimizer.zero_grad()
            predictions = model(user_batch, item_batch, content_batch)
            loss = loss_fn(predictions, rating_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ", leave=False)
        with torch.no_grad():
            for user_batch, item_batch, content_batch, rating_batch in val_bar:
                user_batch = user_batch.to(device)
                item_batch = item_batch.to(device)
                content_batch = content_batch.to(device)
                rating_batch = rating_batch.to(device)

                predictions = model(user_batch, item_batch, content_batch)
                val_loss = loss_fn(predictions, rating_batch)
                total_val_loss += val_loss.item()
                val_bar.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch}/{end_epoch} — Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        torch.save(model.state_dict(), f"data/ncf_model_epoch_{epoch}.pt")

# Start training
continue_training(start_epoch=checkpoint_epoch + 1, end_epoch=end_epoch)
