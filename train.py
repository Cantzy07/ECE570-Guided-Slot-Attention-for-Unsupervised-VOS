import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from load_DAVIS16 import get_davis_dataloader
# Import the model
from models.model import Model 

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the dataset and DataLoader (batch_size=1 due to processor requirements)
    # Create the DataLoader
    train_loader = get_davis_dataloader(
        root_dir="datasets/DAVIS",
        subset="train",
    )
    
    # Instantiate the model and send it to the device.
    model = Model().to(device)
    model.train()  # set to training mode

    # Use CrossEntropyLoss (the model outputs a probability distribution over 3 classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (target_image, reference_images, mask) in enumerate(train_loader):
            # The model's processor handles PIL images, so we do not convert target/reference images.
            # Move mask to the device and add batch dimension if necessary.
            mask = mask.to(device)
            mask = mask.unsqueeze(0)  # [1, H, W]
            
            optimizer.zero_grad()
            
            # Forward pass.
            # Since batch_size=1, target_image and reference_images are lists of one element.
            # We pass the first (and only) element in each.
            output = model(target_image, reference_images)
            # Output is expected to have shape [1, num_classes, H_out, W_out]
            
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {epoch_loss:.4f}")
        
        # Save a checkpoint after each epoch
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the test DataLoader from the DAVIS validation split.
    test_loader = get_davis_dataloader(
        root_dir="datasets/DAVIS",
        subset="val",
    )
    
    # Instantiate the model and move to the device.
    model = Model().to(device)
    
    # Load the checkpoint if a checkpoint path is provided.
    # Make sure to add '--checkpoint_path' as an argument when running the test.
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    else:
        print("No checkpoint provided or found, evaluating untrained model.")
    
    model.eval()  # set the model to evaluation mode
    
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for target_image, reference_images, mask in test_loader:
            # Transfer the ground-truth mask to device.
            mask = mask.to(device).long()
            # In our training function we add an extra batch dimension to the mask
            # since our model expects [B, H, W]; adjust accordingly.
            mask = mask.unsqueeze(0)  # shape: [1, H, W]
            
            # Forward pass: our dataloader returns lists with one element.
            # Pass the first element of the target and reference image lists.
            output = model(target_image, reference_images)
            # The output shape is assumed to be [1, num_classes, H, W]
            
            # Get the predicted class per pixel.
            predicted = torch.argmax(output, dim=1)  # shape: [1, H, W]
            
            # Calculate the number of correctly predicted pixels.
            correct = (predicted == mask).sum().item()
            total_correct += correct
            total_pixels += mask.numel()
    
    pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    print(f"Test Pixel Accuracy: {pixel_accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the segmentation model")
    parser.add_argument("--images_dir", type=str, default="data/images", help="Directory containing target images")
    parser.add_argument("--masks_dir", type=str, default="data/masks", help="Directory containing ground truth masks")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (in steps)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()
    
    train(args)