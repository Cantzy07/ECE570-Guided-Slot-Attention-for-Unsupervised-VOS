import torch.nn as nn
import torch.nn.functional as F

class FAT(nn.Module):
    def __init__(self, local_channels, global_channels, embed_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Convolutional projections (preserve spatial dimensions)
        self.local_conv = nn.Conv2d(local_channels, embed_dim, kernel_size=1)
        self.global_conv = nn.Conv2d(global_channels, embed_dim, kernel_size=1)
        
        # Cross-attention: using standard multihead attention.
        # We'll flatten the spatial dimensions of the local features.
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Feed-forward network: implemented with 1x1 convolutions
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)
        )
        
        # Two normalization layers after attention and feed-forward blocks.
        # (LayerNorm is applied on the channel dimension after reshaping.)
        self.norm1 = nn.LayerNorm(embed_dim)
    
    def forward(self, x_l, x_g):
        B = x_l.shape[0]
        
        # Project local features with a convolution that preserves spatial dimensions.
        local_feat = self.local_conv(x_l)  # [B, embed_dim, H_l, W_l]
        # Flatten the spatial dimensions for attention;
        # shape: [H_l*W_l, B, embed_dim]
        B, C, H_l, W_l = local_feat.shape
        local_flat = local_feat.view(B, C, H_l * W_l).permute(2, 0, 1)

        # Here, we reduce the spatial dimensions (e.g., to 32x32 = 1024 tokens).
        downsampled_feat = F.adaptive_avg_pool2d(local_feat, (32, 32))
        B, C, H_down, W_down = downsampled_feat.shape  # Now H_down*W_down tokens
        local_flat = downsampled_feat.view(B, C, H_down * W_down).permute(2, 0, 1)  # [H_down*W_down, B, embed_dim]
        
        # Process global features with a convolution and apply pooling to get a global summary.
        global_feat = self.global_conv(x_g)  # [B, embed_dim, H_g, W_g]
        # Use adaptive average pooling over the spatial dimensions to retain contextual info.
        global_pooled = F.adaptive_avg_pool2d(global_feat, (1, 1))  # [B, embed_dim, 1, 1]
        global_flat = global_pooled.view(B, 1, self.embed_dim).permute(1, 0, 2)  # [1, B, embed_dim]
        
        # Cross-attention: let each spatial location in the local feature attend to the global context.
        cross_attn_out, _ = self.cross_attn(query=local_flat, key=global_flat, value=global_flat)
        # Residual connection: add cross-attended information back to the original local tokens.
        local_cross = local_flat + cross_attn_out
        
        # Self-attention on local tokens over their spatial grid.
        self_attn_out, _ = self.self_attn(query=local_cross, key=local_cross, value=local_cross)
        local_sa = local_cross + self_attn_out
        
        # Reshape back into spatial dimensions.
        local_sa = local_sa.permute(1, 2, 0).contiguous().view(B, self.embed_dim, H_down, W_down)
        
        # Feed-forward network (with conv layers)
        ffn_out = self.ffn(local_sa)
        ffn_res = local_sa + ffn_out
        
        # Apply normalization.
        # LayerNorm expects the normalized dimension to be the last dimension.
        # Rearrange to [B, H_l, W_l, embed_dim], normalize, then go back.
        norm_input = ffn_res.permute(0, 2, 3, 1).contiguous()
        normed = self.norm1(norm_input)

        aggregated_features = normed.permute(0, 3, 1, 2)
        
        return aggregated_features
