from models.model import SlotGenerator
from test_encoder import encode_image
import torch.nn as nn
from PIL import Image

class SlotGenerator(nn.Module):
    def __init__(self, in_channels, slot_num):
        super(SlotGenerator, self).__init__()  
        self.input_1x1conv = nn.Conv2d(in_channels=in_channels, out_channels=slot_num, kernel_size=1)
        # Pixel-wise softmax
        self.softmax_w = nn.Softmax(dim=-1)  # Apply softmax along the last dimension (width)
        self.softmax_h = nn.Softmax(dim=-2)  # Apply softmax along the second last dimension (height)
        self.globalavgpool = nn.AvgPool2d(kernel_size=(128, 128))  # Pool over the entire spatial dimension

    def forward(self, x):
        x = self.input_1x1conv(x)
        x = self.softmax_w(x)
        x = self.softmax_h(x)
        x = self.globalavgpool(x)
        return x

def generate_slots(image):
    stage_1, stage_2, stage_3, stage_4 = encode_image(image)

    slot_generator = SlotGenerator(in_channels=64, slot_num=2)

    slots = slot_generator(stage_1)

    print(f"Slot Output Shape: {slots.shape}")

    return slots

if __name__ == "__main__":
    image_path = "C:\\School\\Y3\\SP25\\ECE570\\ECE570-Guided-Slot-Attention-for-Unsupervised-VOS\\datasets\\DAVIS\\JPEGImages\\1080p\\dog\\00000.jpg"
    image = Image.open(image_path)
    generate_slots(image)

    # Input: Stage 1 (Local) Shape: torch.Size([1, 64, 128, 128])
    # Output: Slot Output Shape: torch.Size([1, 2, 1, 1])