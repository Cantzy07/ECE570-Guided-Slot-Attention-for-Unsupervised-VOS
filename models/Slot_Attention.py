import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedSlotAttention(nn.Module):
    def __init__(self, embed_dim=256, num_slots=2, num_iters=3, num_knn=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.num_knn = num_knn

        # Enhanced slot initialization
        self.slot_proj = nn.Sequential(
            nn.Conv2d(num_slots, num_slots * embed_dim, kernel_size=1),  # [B,2,H,W] -> [B,256,H,W]
            nn.ReLU(),
            nn.Conv2d(num_slots * embed_dim, num_slots * embed_dim, kernel_size=1)   # [B,256,H,W]
        )

        # Feature projection to match slot dimension
        self.feature_proj = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=1),  # First expand to 256 channels
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)  # Then maintain dimension
        )

        # Slot update mechanism
        self.update_gru = nn.GRUCell(embed_dim, embed_dim)

    def _get_knn_features(self, slots, features):
        """Compute KNN features with proper tensor shapes"""
        B, C, H, W = features.shape
        
        # 1. Normalize and reshape slots [B, num_slots, C] -> [B, num_slots, C]
        slots_norm = F.normalize(slots, p=2, dim=-1)
        
        # 2. Prepare features [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        features_flat = features.flatten(2).permute(0, 2, 1)
        features_norm = F.normalize(features_flat, p=2, dim=-1)
        
        # 3. Compute similarity [B, num_slots, H*W]
        sim_matrix = torch.matmul(
            slots_norm,  # [B, num_slots, C]
            features_norm.transpose(-1, -2)  # [B, C, H*W]
        )
        
        # 4. Get top-K features
        _, topk_indices = torch.topk(sim_matrix, self.num_knn, dim=2)  # [B, num_slots, K]
        
        # 5. Gather features
        knn_features = torch.gather(
            features_flat.unsqueeze(1).expand(-1, self.num_slots, -1, -1),
            dim=2,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
        )  # [B, num_slots, K, C]
        
        return knn_features

    def forward(self, PA, PS_init):     
        # Project features to match slot dimension
        PA = self.feature_proj(PA)  # Expected shape: [B, 256, H, W]
        
        # Initialize slots with shape: [B, num_slots*embed_dim, H, W]
        slots = self.slot_proj(PS_init)  
        B, C, H, W = slots.shape  # Here, C should be num_slots * embed_dim
        
        # Reshape slots to separate the channels for each slot:
        slots = slots.view(B, self.num_slots, self.embed_dim, H, W)  # Shape: [B, num_slots, embed_dim, H, W]
        
        # Pool spatially (e.g., global average pooling) to obtain one vector per slot:
        slots = slots.mean(dim=[-2, -1])  # Final shape: [B, num_slots, embed_dim]
        
        for _ in range(self.num_iters):
            # 1. KNN selection (using cosine similarity)
            knn_features = self._get_knn_features(slots, PA)  # Expected shape: [B, num_slots, num_knn, embed_dim]
            
            updated_slots = []
        for slot_idx in range(self.num_slots):
            # Compute attention weights
            query = slots[:, slot_idx].unsqueeze(1)  # [B, 1, C]
            keys = knn_features[:, slot_idx]         # [B, K, C]
            
            # Scaled dot-product attention
            attn_logits = (query @ keys.transpose(1, 2)) / (self.embed_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=-1)     # [B, 1, K]
            
            # Weighted feature aggregation
            aggregated = (attn @ keys).squeeze(1)       # [B, C]
            
            # Update the slot using GRUCell (without in-place modification)
            updated_slot = self.update_gru(aggregated, slots[:, slot_idx])
            updated_slots.append(updated_slot)

        # Reassemble the slots tensor from the updated slots list
        slots = torch.stack(updated_slots, dim=1)  # [B, num_slots, C]
        
        return slots  # [B, num_slots, embed_dim]
    