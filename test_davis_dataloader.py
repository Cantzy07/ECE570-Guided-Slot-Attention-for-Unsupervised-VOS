import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Import the get_davis_dataloader function.
# Adjust the import statement depending on your module structure.
# For example, if your DAVIS dataset code is saved as davis_dataset.py:
from load_DAVIS16 import get_davis_dataloader

def display_sample(target_image, reference_images, target_annotation):
    """
    Display the target image, target annotation, and all reference images in a more
    readable layout: one row for the target image & annotation, followed by a grid 
    for the references.
    """
    import math
    import matplotlib.pyplot as plt

    # Convert target image and annotation to numpy.
    target_np = target_image.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    annotation_np = target_annotation.detach().cpu().numpy()           # (H, W)

    # --- First Figure Row for Target & Annotation ---
    # We'll create a "gridspec" with 2 rows:
    #   Row 1 => 2 columns (Target Image & Annotation)
    #   Row 2 => references
    fig = plt.figure(figsize=(12, 6))
    # Create a grid of 2 rows. The first row will have 2 columns, the second row we'll fill with references.
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3])  # first row is smaller, second row is bigger

    # Target Image in (row=0, col=0)
    ax_target = fig.add_subplot(gs[0, 0])
    ax_target.imshow(target_np)
    ax_target.set_title('Target Image')
    ax_target.axis('off')

    # Annotation in (row=0, col=1)
    ax_anno = fig.add_subplot(gs[0, 1])
    ax_anno.imshow(annotation_np, cmap='gray')
    ax_anno.set_title('Target Annotation')
    ax_anno.axis('off')

    # --- Second Figure Row for References in a grid ---
    # We want all references in row=1, but many columns. We'll create sub-gridspec based on how many references we have.
    n_refs = len(reference_images)
    # Decide how many columns you want for references:
    n_cols = 5  # 5 references per row, for example
    n_rows = math.ceil(n_refs / n_cols)
    
    # Create a new gridspec just for references in the bottom row. 
    # We tell it "row=1, col=0:2" to span both columns in the second row.
    gs_refs = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[1, :], wspace=0.05, hspace=0.2)

    reference_images = reference_images[:1]

    for i, ref in enumerate(reference_images):
        # Convert each reference image from shape (1, 3, H, W) to (3, H, W), then (H, W, 3).
        ref_3d = ref.squeeze(0)  # -> shape [3, H, W]
        ref_np = ref_3d.detach().cpu().numpy().transpose(1, 2, 0)  # -> shape (H, W, 3)

        # Compute row & col within this sub-gridspec.
        row = i // n_cols
        col = i % n_cols

        ax_ref = fig.add_subplot(gs_refs[row, col])
        ax_ref.imshow(ref_np)
        ax_ref.axis('off')
        ax_ref.set_title(f'Ref {i+1}')

    plt.tight_layout()
    plt.show()

def test_get_davis_dataset():
    """
    Tests the get_davis_dataloader by loading one batch and printing the sizes
    and number of reference images. It also visualizes one sample.
    """
    # Define simple transformations. Here, we use ToTensor() to convert PIL images to tensors.
    transform = T.ToTensor()
    target_transform = None  # Let the dataset's default annotation conversion handle it.

    # Create the dataloader for training.
    # Adjust root_dir if necessary. The selected_titles parameter can be used to limit the dataset, e.g., only "bear".
    dataloader = get_davis_dataloader(
        root_dir="datasets/DAVIS",
        subset="train",
        resolution="480p",
        transform=transform,
        target_transform=target_transform,
        selected_titles=None,  # Remove or change as needed.
        batch_size=1,             # Using batch_size=1 to keep the reference image list intact.
        shuffle=False,
        num_workers=0             # Set to 0 for testing to avoid multiprocessing complications.
    )

    dataset_length = len(dataloader.dataset)
    print(f"Total number of samples in dataset: {dataset_length}")

    # Iterate over one batch (batch_size=1)
    # Note: Since reference images are returned as a list, the default collate_fn will wrap it in a list (batch dimension).
    for batch in dataloader:
        # Unpack the batch. With batch_size=1, these will be single-element lists/tensors.
        target_image, reference_images, target_annotation = batch

        # Print out the shapes. target_image and target_annotation are batched tensors.
        print("Target Image tensor shape:", target_image.shape)         # Expected shape: [1, C, H, W]
        print("Target Annotation tensor shape:", target_annotation.shape)   # Expected shape: [1, H, W]

        # reference_images is a list of lists; extract the list for our batch.
        print("Number of Reference Images for this sample:", len(reference_images))

        # Optionally, visualize the sample.
        # Squeeze the batch dimension from the target image and annotation.
        display_sample(target_image.squeeze(0), reference_images, target_annotation.squeeze(0))

if __name__ == '__main__':
    test_get_davis_dataset()