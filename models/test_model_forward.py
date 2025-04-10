from model import Model
import torch

import torch
from PIL import Image
from transformers import SegformerForImageClassification, SegformerImageProcessor
import torch.nn.functional as F
import random
import os

def encode_image(image):
    rgb_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = rgb_encoder(**inputs, output_hidden_states=True)

    # Outputs contain multi-scale feature maps
    # B, C, H, W
    stage_1 = outputs.hidden_states[0]  # Local (1/4 size)
    stage_2 = outputs.hidden_states[1]  # Local (1/8 size)
    stage_3 = outputs.hidden_states[2]  # Global (1/16 size)
    stage_4 = outputs.hidden_states[3]  # Global (1/32 size)

    print(f"Stage 1 (Local) Shape: {stage_1.shape}")
    print(f"Stage 2 (Local) Shape: {stage_2.shape}")
    print(f"Stage 3 (Global) Shape: {stage_3.shape}")
    print(f"Stage 4 (Global) Shape: {stage_4.shape}")

    return stage_1, stage_2, stage_3, stage_4

def test_global_extraction(data_dir):
    # images is a batch of reference images
    rgb_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")

    # Prepare a list to hold the features from each image
    feature_maps = []

    num_images = 5
    images = load_images_from_folder(data_dir, num_images)

    # Loop through each image
    for image in images:
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Forward pass through the model
        with torch.no_grad():
            outputs = rgb_encoder(**inputs, output_hidden_states=True)

        # Outputs contain multi-scale feature maps
        stage_1 = outputs.hidden_states[0]  # Local (1/4 size)
        stage_2 = outputs.hidden_states[1]  # Local (1/8 size)
        stage_3 = outputs.hidden_states[2]  # Global (1/16 size)
        stage_4 = outputs.hidden_states[3]  # Global (1/32 size)

        # Apply channel-wise softmax: softmax across the channel dimension (dim=0)
        softmax_feature_map = F.softmax(stage_4, dim=1)  # Softmax across channels
        
        # Reduce the channels to a single channel by summing them
        mj_gt = softmax_feature_map.sum(dim=1, keepdim=True)  # Sum over channels, keeping the spatial dimensions

        # You can choose which stage to use, here I am using stage_1 (local, 1/4 size)
        feature_maps.append(mj_gt)  # Add the feature map to the list

    # Stack the feature maps along the channel dimension
    # Assuming feature_maps is a list of tensors with shape [B, C, H, W]
    combined_features = torch.cat(feature_maps, dim=1)  # Concatenate along the channel dimension

    print(f"Combined Features Shape: {combined_features.shape}")

    return combined_features

def load_images_from_folder(image_dir, num_images):
    # Get a list of all images in the directory (support .jpg, .png, .jpeg)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly select a subset of images
    selected_files = random.sample(image_files, min(num_images, len(image_files)))

    images = []
    for image_file in selected_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        images.append(image)

    return images

if __name__ == "__main__":
    image_path = "C:\\School\\Y3\\SP25\\ECE570\\ECE570-Guided-Slot-Attention-for-Unsupervised-VOS\\datasets\\DAVIS\\JPEGImages\\1080p\\dog\\00000.jpg"
    target = Image.open(image_path)

    data_dir = "C:\\School\\Y3\\SP25\\ECE570\\ECE570-Guided-Slot-Attention-for-Unsupervised-VOS\\datasets\\DAVIS\\JPEGImages\\1080p\\dog"
    num_images = 5
    references = load_images_from_folder(data_dir, num_images)

    model = Model()
    mask = model(target, references)
    print("mask shape", mask.shape)

    # mask shape torch.Size([1, 258, 32, 32])