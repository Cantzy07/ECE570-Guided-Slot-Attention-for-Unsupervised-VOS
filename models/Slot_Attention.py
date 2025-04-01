import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from sklearn.neighbors import NearestNeighbors  # For KNN

class GuidedSlotAttention(nn.Module):
    def __init__(self, embed_dim=256, num_slots=2, num_iters=3, num_knn=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.num_knn = num_knn

        # Projection for initial slots
        self.slot_proj = nn.Linear(1, embed_dim)  # [1, 2, 1, 1] -> [1, 2, embed_dim]

        # FAT for slot-feature attention (simplified)
        self.fat = MultiheadAttention(embed_dim, num_heads=4)

        # Slot update GRU (optional)
        self.gru = nn.GRUCell(embed_dim, embed_dim)

        # Attentive pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, PA, PS_init):
        # PA: [1, 1, 16, 16] -> [1, 256]
        PA_flat = PA.view(1, -1).unsqueeze(0)  # [1, 1, 256]
        
        # Initialize slots: [1, 2, 1, 1] -> [1, 2, embed_dim]
        slots = self.slot_proj(PS_init.view(1, 2, 1)).squeeze(-1)  # [1, 2, embed_dim]

        for _ in range(self.num_iters):
            # Step 1: KNN Filtering (select N nearest features per slot)
            knn_indices = []
            for slot in slots[0]:  # For each slot (2 total)
                dists = torch.norm(PA_flat - slot, dim=-1)  # L2 distance
                _, indices = torch.topk(dists, self.num_knn, largest=False)
                knn_indices.append(indices)
            knn_indices = torch.stack(knn_indices)  # [2, N]

            # Step 2: FAT Attention (slots attend to KNN-filtered features)
            knn_features = PA_flat[:, knn_indices]  # [1, 2, N, embed_dim]
            knn_features = knn_features.view(1, 2 * self.num_knn, -1)  # [1, 2*N, embed_dim]
            
            # FAT attention (slots as queries, KNN features as keys/values)
            slots = slots.transpose(0, 1)  # [2, 1, embed_dim] for MHA
            attn_out, _ = self.fat(
                query=slots,
                key=knn_features,
                value=knn_features
            )  # [2, 1, embed_dim]
            slots = attn_out.transpose(0, 1)  # [1, 2, embed_dim]

            # Step 3: Slot Update (GRU or MLP)
            slots = self.gru(
                slots.view(-1, self.embed_dim),
                slots.view(-1, self.embed_dim)
            ).view(1, 2, -1)

        # Attentive Pooling (optional)
        slot_weights = self.attn_pool(slots)  # [1, 2, 1]
        refined_slots = (slots * slot_weights).sum(dim=1)  # [1, embed_dim]

        return refined_slots