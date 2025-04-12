from models.model import SlotGenerator
from module_tests.test_encoder import encode_image
import torch.nn as nn
from PIL import Image
import os 

def generate_slots(image):
    stage_1, stage_2, stage_3, stage_4 = encode_image(image)

    slot_generator = SlotGenerator(in_channels=64, slot_num=2)

    slots = slot_generator(stage_1)

    print(f"Slot Output Shape: {slots.shape}")

    return slots

if __name__ == "__main__":
    image_path = os.path.join("datasets", "DAVIS", "JPEGImages", "1080p", "dog", "00000.jpg")
    image = Image.open(image_path)
    generate_slots(image)

    # Input: Stage 1 (Local) Shape: torch.Size([1, 64, 128, 128])
    # Output: Slot Output Shape: torch.Size([1, 2, 128, 128])