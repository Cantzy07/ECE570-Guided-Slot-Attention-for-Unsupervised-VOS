from models.feature_aggregation_transformer import FAT
import torch

# Modified forward method to debug shapes at each step
def trace_forward(model, x_l, x_g):
    print(f"Input x_l shape: {x_l.shape}")
    print(f"Input x_g shape: {x_g.shape}")
    
    x_lv = model.plv(x_l)
    print(f"x_lv shape after plv: {x_lv.shape}")
    
    x_lq = x_lv
    x_lk = model.plk(x_l)
    print(f"x_lk shape after plk: {x_lk.shape}")
    
    try:
        x_gq = model.pgq(x_g)
        print(f"x_gq shape after pgq: {x_gq.shape}")
        
        x_lg_kq = torch.matmul(x_lk, x_gq.transpose(-2, -1))
        print(f"x_lg_kq shape after matmul: {x_lg_kq.shape}")
        
        # Softmax2d expects 4D input (N,C,H,W)
        # This needs to be reshaped or use a different softmax
        print("Note: nn.Softmax2d() expects 4D input (N,C,H,W) - consider using nn.Softmax(dim=-1) instead")
        
        # Continue with suggested corrections...
        
    except Exception as e:
        print(f"Error during tracing: {e}")

if __name__ == "__main__":

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # local features: torch.Size([1, 64, 128, 128])
    # global features: torch.Size([1, batch_size, 16, 16])

    # Test parameters
    batch_size = 4
    embedding_size = 32     # Embedding size for the transformer

    # Create artificial input tensors
    x_l = torch.randn(1, 64, 128, 128)  # [batch_size, 64, 128, 128]
    x_g = torch.randn(1, batch_size, 16, 16)  # [batch_size, batch_size, 16, 16]

    # Initialize the model
    model = FAT(local_channels=64, global_channels=batch_size)

    # Try to do a forward pass
    output = model(x_l, x_g)
    print(f"Forward pass successful!")
    print(f"Input shapes: x_l = {x_l.shape}, x_g = {x_g.shape}")
    print(f"Output shape: {output.shape}")
    # [1, 256, 32, 32]