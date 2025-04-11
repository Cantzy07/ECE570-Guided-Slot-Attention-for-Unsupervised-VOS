import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def visualize_mask(mask):
    """
    Visualize the predicted mask.
    
    Args:
        mask (torch.Tensor): Output mask from the model with shape [1, 3, 128, 128]
                             where each channel is the predicted probability for a class.
    """
    # Remove the batch dimension and get predictions via argmax over the channel dimension.
    # This produces a tensor of shape [128, 128] with integer labels (0, 1, or 2).
    pred_labels = mask.argmax(dim=1).squeeze(0)
    
    # Convert to numpy for plotting.
    pred_np = pred_labels.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    # Use a colormap (e.g., 'jet' or 'viridis') for visualization.
    plt.imshow(pred_np, cmap='jet', interpolation='nearest')
    plt.title("Predicted Mask (Argmax over Channels)")
    plt.axis('off')
    plt.show()

def overlay_mask_on_target(target_image, mask, alpha=0.5):
    """
    Overlay the predicted mask on the target image.
    
    Args:
        target_image (torch.Tensor): Target image tensor of shape [1, 3, H, W].
        mask (torch.Tensor): Output mask from the model with shape [1, 3, H, W].
        alpha (float): Transparency factor for the overlay.
    """
    # Get the predicted label map.
    pred_labels = mask.argmax(dim=1).squeeze(0)  # shape: [H, W]
    
    # Convert target image to a numpy array for display.
    # Assuming target_image is in [0,1] range.
    image_np = target_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # Create a color map overlay for the mask.
    # Here, for simplicity, we convert the predicted labels to an RGB image using a colormap.
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.imshow(pred_labels.detach().cpu().numpy(), cmap='jet', alpha=alpha)
    plt.title("Overlay: Predicted Mask on Target Image")
    plt.axis('off')
    plt.show()


# Example usage:
if __name__ == '__main__':
    # For demonstration, create dummy inputs.
    # Let's assume your model's output mask is of shape [1, 3, 128, 128].
    # Here, we simulate this with random probabilities.
    dummy_mask = torch.rand(1, 3, 128, 128)
    
    # And assume a target image tensor of shape [1, 3, 128, 128] (normalized to [0,1]).
    dummy_target = torch.rand(1, 3, 128, 128)
    
    # Visualize the predicted mask by itself.
    visualize_mask(dummy_mask)
    
    # Visualize an overlay of the mask on the target image.
    overlay_mask_on_target(dummy_target, dummy_mask, alpha=0.5)