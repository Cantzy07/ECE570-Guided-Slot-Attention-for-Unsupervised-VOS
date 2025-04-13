import argparse
import os
import torch
from load_DAVIS16 import get_davis_dataloader
# Import the model
from models.model import Model 
import torchvision.transforms as T
import numpy as np
from skimage import feature, morphology

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Create the test DataLoader using the DAVIS validation split.
test_loader = get_davis_dataloader(
    root_dir="datasets/DAVIS",
    subset="val",
    resolution="480p",
    transform=T.ToTensor(),
    target_transform=None,
    selected_titles=None,   # Modify as needed.
    batch_size=1,           # Using batch_size=1 to ensure reference image list is intact.
    shuffle=False,
    num_workers=0           # Set to 0 for simpler debugging and evaluation.
)

# Instantiate the model and move it to the device.
model = Model(local_channels=64, global_channels=5, embed_dim=256, num_slots=2).to(device)

def test(args):
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
            # The model's processor handles PIL images, so we do not convert target/reference images.
            # Move mask to the device and add batch dimension if necessary.
            mask = mask.to(device)
            mask = (mask > 0).long()  # Convert 255→1, keep 0→0

            if mask.dim() == 3:
                # Case 1: shape is [480, 854, 2] (or similar) -> one-hot encoding.
                if mask.shape[-1] == 2:
                    # Convert one-hot encoding to single channel by taking the argmax.
                    mask = mask.argmax(dim=-1)  # Now shape [480, 854]
                # Now if the first dimension is not a batch dim, add one:
                if mask.shape[0] != 1:
                    mask = mask.unsqueeze(0)  # Now shape [1, 480, 854]
            elif mask.dim() == 2:
                # Case 2: shape is [480, 854] -> add batch dimension.
                mask = mask.unsqueeze(0)  # Now shape [1, 480, 854]
            
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


def jaccard_index(args):
    # Load the checkpoint if provided.
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    else:
        print("No checkpoint provided or found, evaluating untrained model.")
    
    model.eval()  # Set the model to evaluation mode.
    
    total_intersection = 0
    total_union = 0

    with torch.no_grad():
        for target_image, reference_images, mask in test_loader:
            # Move mask to the device and convert to a binary mask.
            mask = mask.to(device)
            mask = (mask > 0).long()  # This converts values > 0 to 1 (foreground), keeping 0 as background.

            # Adjust tensor dimensions if necessary.
            if mask.dim() == 3:
                # Case: shape is [H, W, 2] (one-hot) or [H, W, ?]
                if mask.shape[-1] == 2:
                    mask = mask.argmax(dim=-1)  # Now shape [H, W].
                if mask.shape[0] != 1:
                    mask = mask.unsqueeze(0)  # Now shape [1, H, W].
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0)  # Now shape [1, H, W].

            # Forward pass: our dataloader returns lists with one element each.
            output = model(target_image, reference_images)
            # Output shape is assumed to be [1, num_classes, H, W].
            
            # Get the predicted class per pixel.
            predicted = torch.argmax(output, dim=1)  # Shape: [1, H, W].

            # For binary segmentation, compute the intersection and union for the foreground class (1).
            # Intersection: pixels that are both predicted and in the ground truth.
            # Union: pixels that are predicted OR in the ground truth.
            intersection = ((predicted == 1) & (mask == 1)).sum().item()
            union = ((predicted == 1) | (mask == 1)).sum().item()
            
            total_intersection += intersection
            total_union += union

    # Avoid division by zero: if there is no foreground in both masks, define IoU as 1.
    if total_union == 0:
        jaccard = 1.0
    else:
        jaccard = total_intersection / total_union

    print(f"Test Jaccard Index: {jaccard * 100:.2f}%")

def boundary_measure(args, tolerance=2):
    # Load the checkpoint if provided.
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    else:
        print("No checkpoint provided or found, evaluating untrained model.")
    
    model.eval()  # set the model to evaluation mode
    
    def compute_boundary_f_measure(gt_mask, pred_mask, tol):
        """
        Compute the boundary F-measure between a ground truth mask and a predicted mask.
        Both inputs should be binary numpy arrays.
        """
        # Extract boundaries using the Canny edge detector.
        # Here, we convert the binary masks to float (0.0 or 1.0) for processing.
        gt_edge = feature.canny(gt_mask.astype(float))
        pred_edge = feature.canny(pred_mask.astype(float))
        
        # Dilate the ground truth boundary to allow for small misalignments.
        gt_edge_dilated = morphology.binary_dilation(gt_edge, morphology.disk(tol))
        
        # True positives: predicted boundary pixels that fall inside the dilated ground truth boundary.
        true_positive = np.logical_and(pred_edge, gt_edge_dilated).sum()
        
        # Precision: Ratio of correctly predicted boundary pixels to total predicted boundary pixels.
        precision = true_positive / (pred_edge.sum() + 1e-8)
        # Recall: Ratio of correctly predicted boundary pixels to total ground truth boundary pixels.
        recall = true_positive / (gt_edge.sum() + 1e-8)
        
        if precision + recall == 0:
            return 0.0
        f_measure = 2 * precision * recall / (precision + recall)
        return f_measure

    total_f_measure = 0.0
    num_samples = 0

    with torch.no_grad():
        for target_image, reference_images, mask in test_loader:
            # Move mask to the device and convert to binary format (threshold > 0 becomes foreground).
            mask = mask.to(device)
            mask = (mask > 0).long()
            
            # Ensure mask has proper dimensions.
            if mask.dim() == 3:
                if mask.shape[-1] == 2:  # if one-hot encoded (e.g., 2 channels)
                    mask = mask.argmax(dim=-1)
                if mask.shape[0] != 1:
                    mask = mask.unsqueeze(0)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0)
            
            # Forward pass: use the first element from the lists returned by the dataloader.
            output = model(target_image, reference_images)
            # Assume output shape [1, num_classes, H, W]
            predicted = torch.argmax(output, dim=1)  # shape: [1, H, W]

            # Convert predictions and ground truth to numpy arrays.
            pred_np = predicted.cpu().numpy()[0]
            gt_np = mask.cpu().numpy()[0]
            
            # For binary segmentation, consider the foreground as the class "1".
            # If your task involves multi-class, you may want to compute the metric per class.
            pred_binary = (pred_np == 1).astype(np.uint8)
            gt_binary = (gt_np == 1).astype(np.uint8)
            
            # Compute the boundary F-measure for this sample.
            sample_f = compute_boundary_f_measure(gt_binary, pred_binary, tolerance)
            total_f_measure += sample_f
            num_samples += 1
    
    # Average boundary F-measure over all samples.
    avg_boundary_f = total_f_measure / num_samples if num_samples > 0 else 0.0
    print(f"Test Boundary F-measure: {avg_boundary_f * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the segmentation model")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_epoch_5.pth", help="Directory to save model checkpoints")
    args = parser.parse_args()

    # test(args)
    
    I_m = jaccard_index(args)
    F_m = boundary_measure(args)
    G_m = (I_m + F_m) / 2

    print("G_m I_m F_m", G_m, I_m, F_m)