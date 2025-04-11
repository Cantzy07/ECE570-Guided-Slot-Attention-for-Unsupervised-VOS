from feature_aggregation_transformer import FAT
from Slot_Attention import GuidedSlotAttention
from test_slotgenerator import generate_slots
import torch
from PIL import Image

if __name__ == "__main__":

    # Set random seed for reproducibility
    torch.manual_seed(42)

    image_path = "C:\\School\\Y3\\SP25\\ECE570\\ECE570-Guided-Slot-Attention-for-Unsupervised-VOS\\datasets\\DAVIS\\JPEGImages\\1080p\\dog\\00000.jpg"
    image = Image.open(image_path)
    slots = generate_slots(image)

    # local features: torch.Size([1, 64, 128, 128])
    # global features: torch.Size([1, batch_size, 16, 16])

    # Test parameters
    batch_size = 4
    embedding_size = 32     # Embedding size for the transformer

    # Create artificial input tensors
    x_l = torch.randn(1, 64, 128, 128)  # [batch_size, 64, 128, 128]
    x_g = torch.randn(1, batch_size, 16, 16)  # [batch_size, batch_size, 16, 16]

    # Initialize the model
    gsa = GuidedSlotAttention()
    fat = FAT(local_in=128, global_in=16)
    fat_output = fat(x_l, x_g)
    slotatt_output = gsa(fat_output, slots)

    print("gsa shape", slotatt_output.shape)
    