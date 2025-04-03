import torch.nn as nn
import torch
import math

class AttentivePooling(nn.Module):
    """Pool global features into a compact vector."""
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, 1)  # Learned attention mechanism

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        attn_weights = torch.softmax(self.query(x), dim=1)  # [batch, seq_len, 1]
        pooled = torch.sum(x * attn_weights, dim=1)  # [batch, embed_dim]
        return pooled

class CrossAttention(nn.Module):
    """Local features (query) attend to global features (key, value)."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_local, x_global):
        # x_local: [batch, seq_len, embed_dim] (queries)
        # x_global: [batch, embed_dim] (keys/values)
        batch_size, seq_len, _ = x_local.shape

        # Project queries (local)
        q = self.q_proj(x_local).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]

        # Project keys/values (global)
        kv = self.kv_proj(x_global).view(batch_size, 1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [batch, heads, 1, head_dim]

        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale  # [batch, heads, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)  # [batch, seq_len, embed_dim]

        return self.out_proj(output)

class FAT(nn.Module):
    def __init__(self, local_in, global_in, embed_dim=256, num_heads=4):
        super().__init__()
        # Local feature projection
        self.local_proj = nn.Linear(local_in*local_in, embed_dim)

        # Global feature projection + pooling
        self.global_proj = nn.Linear(global_in*global_in, embed_dim)
        self.attentive_pool = AttentivePooling(embed_dim)

        # Cross-attention (local × global)
        self.cross_attn = CrossAttention(embed_dim, num_heads)

        # Self-attention + FFN
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Mask decoder (example)
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)  # Output mask [1, H, W]
        )
    
    def forward(self, x_l, x_g):
        # Local features: [1, 64, 128, 128] → [1, 64, embed_dim]
        x_l_flat = x_l.view(1, 64, -1)
        x_local = self.local_proj(x_l_flat)

        # Global features: [1, batch_size, 16, 16] → [1, embed_dim]
        x_g_flat = x_g.view(1, x_g.shape[1], -1)
        x_global = self.global_proj(x_g_flat)
        x_global = self.attentive_pool(x_global)  # [1, embed_dim]

        # Cross-attention
        x_cross = self.cross_attn(x_local, x_global)  # [1, 64, embed_dim]

        # Self-attention + FFN
        x_self = self.self_attn(x_cross, x_cross, x_cross)[0]  # [1, 64, embed_dim]
        x_out = self.norm1(x_cross + x_self)
        x_out = self.norm2(x_out + self.ffn(x_out))

        # Generate mask: [1, 64, embed_dim] → [1, embed_dim, 8, 8] → upsample
        x_out = x_out.transpose(1, 2).view(1, -1, 8, 8)  # Adjust spatial dims as needed
        mask = self.mask_decoder(x_out)  # [1, 1, H, W]

        return mask