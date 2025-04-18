import torch
import torch.nn as nn

# NCFModel using Collaborative Filtering (Embedding-based approach)
class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, content_dim):
        super(NCFModel, self).__init__()
        
        # Embedding layers for collaborative filtering
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(embedding_dim * 2 + content_dim, 64)  # Concatenated user-item embeddings and content features
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, user_ids, item_ids, content_features):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate user, item embeddings, and content features
        x = torch.cat([user_emb, item_emb, content_features], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x