import argparse
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from matplotlib.colors import ListedColormap

# Import your DAVIS dataloader and Model.
from load_DAVIS16 import get_davis_dataloader
from models.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def demo(args):
    # Create a dataloader for one iteration (batch_size=1)
    dataloader = get_davis_dataloader(
        root_dir="datasets/DAVIS",
        subset="val",           # or use 'train' if preferred
        resolution="480p",
        transform=T.ToTensor(),
        target_transform=None,
        selected_titles=["cows"],   # Only select the "cows" image sequence
        batch_size=1,               # Only one batch is needed for demo
        shuffle=False,
        num_workers=0               # Set to 0 to avoid multiprocessing issues
    )

    # Get one batch (one iteration) from the dataloader.
    # The batch usually contains: target_image, reference_images, and mask.
    batch = next(iter(dataloader))
    target_image, reference_images, gt_mask = batch

    # Instantiate the model and load the checkpoint.
    model = Model(local_channels=64, global_channels=5, embed_dim=256, num_slots=2).to(device)
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    else:
        print("No valid checkpoint provided, using untrained model.")

    model.eval()  # Set model to evaluation mode.
    with torch.no_grad():
        # Run the model on the batch. The model's processor handles input format.
        output = model(target_image, reference_images)

    # The model outputs a tensor of shape [1, 3, 64, 64].
    # Get the predicted segmentation per pixel via argmax.
    predicted = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Convert the target image from tensor to a PIL image for display.
    # Assumes target_image is a tensor of shape [1, C, H, W].
    target_img_disp = to_pil_image(target_image.squeeze(0).cpu())

    # Process the ground truth mask for visualization.
    # In this example, we assume gt_mask is of shape [1, H, W]. Squeeze out the batch dimension.
    gt_mask_disp = gt_mask.squeeze(0).cpu().numpy()

    # Create a figure with 1 row and 3 columns.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Display the target image.
    axes[0].imshow(target_img_disp)
    axes[0].set_title("Target Image")
    axes[0].axis("off")

    # Display the ground truth annotation mask.
    # Using a grayscale colormap; adjust if your annotations have specific classes.
    axes[1].imshow(gt_mask_disp, cmap="gray")
    axes[1].set_title("Annotation Mask")
    axes[1].axis("off")

    # Display the predicted segmentation mask.
    custom_cmap = ListedColormap(['black', 'red'])
    axes[2].imshow(predicted, cmap=custom_cmap)
    axes[2].set_title("Predicted Segmentation Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo using DAVIS dataloader")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/model_epoch_5.pth",
        help="Path to the model checkpoint"
    )
    args = parser.parse_args()
    demo(args)