import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, user_embedding_dim=128, history_dim=128, hidden_dim=256, output_dim=128):
        super(UserTower, self).__init__()
        
        # User ID Embedding (Captures latent user traits)
        # Assuming max 10,000 users for now, can be dynamic
        self.user_embedding = nn.Embedding(10000, user_embedding_dim)
        
        # History Processor (Processes sequence of book embeddings)
        # Input: Sequence of Book Vectors (Batch, Seq_Len, Embed_Dim)
        self.history_gru = nn.GRU(input_size=384, hidden_size=history_dim, batch_first=True)
        
        # Semantic/Explicit Interest Embedding
        # Example: Linear layer to project concatenated search/interest vectors
        self.interest_projector = nn.Linear(384, history_dim)
        
        # Fusion Layer
        self.fc1 = nn.Linear(user_embedding_dim + history_dim + history_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, user_ids, history_vectors, interest_vectors):
        # 1. User ID Embedding
        u_emb = self.user_embedding(user_ids) # (B, user_dim)
        
        # 2. Process History (Sequence of vectors)
        # history_vectors shape: (B, Seq_Len, 768)
        _, h_n = self.history_gru(history_vectors) # h_n: (1, B, history_dim)
        hist_emb = h_n.squeeze(0) # (B, history_dim)
        
        # 3. Process Explicit Interests (Average of vectors)
        # interest_vectors shape: (B, 768)
        int_emb = F.relu(self.interest_projector(interest_vectors)) # (B, history_dim)
        
        # 4. Concatenate
        combined = torch.cat([u_emb, hist_emb, int_emb], dim=1)
        
        # 5. MLP
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Normalize for Cosine Similarity
        return F.normalize(x, p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, book_embedding_dim=384, hidden_dim=256, output_dim=128):
        super(ItemTower, self).__init__()
        
        # Input is pre-computed BERT/Text embedding of the book
        self.fc1 = nn.Linear(book_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, book_vectors):
        # book_vectors shape: (B, 384)
        x = F.relu(self.fc1(book_vectors))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Normalize for Cosine Similarity
        return F.normalize(x, p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, user_config=None, item_config=None):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(**(user_config or {}))
        self.item_tower = ItemTower(**(item_config or {}))
        
    def forward(self, user_input, item_input):
        """
        user_input: tuple (user_ids, history_vectors, interest_vectors)
        item_input: tensor (book_vectors)
        """
        user_embedding = self.user_tower(*user_input)
        item_embedding = self.item_tower(item_input)
        return user_embedding, item_embedding

    def predict_similarity(self, user_input, item_input):
        u, i = self.forward(user_input, item_input)
        return (u * i).sum(dim=1) # Dot product of normalized vectors = Cosine Similarity
