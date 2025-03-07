import torch
from PIL import Image
from transformers import SegformerForImageClassification, SegformerImageProcessor

def encode_image(image):
    rgb_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = rgb_encoder(**inputs)

    # If you want the encoded features (before the classification head)
    encoded_features = outputs.last_hidden_state

    logits = outputs.logits

    print("Logits Tensor Shape", logits.shape)
    print("Encoded Tensor Shape:", encoded_features.shape)
    return encoded_features

if __name__ == "__main__":
    image_path = ""
    encode_image(image_path)