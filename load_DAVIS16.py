import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DAVISDataset(Dataset):
    """
    A Dataset for DAVIS-16 single-object segmentation.
    Reads sequence names from ImageSets/480p/<subset>.txt,
    then collects (image_path, annotation_path) pairs from
    JPEGImages/<resolution>/sequence and Annotations/<resolution>/sequence.
    """
    def __init__(self,
                 root_dir="datasets/DAVIS",
                 subset="train",
                 resolution="480p",
                 transform=None,
                 target_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.resolution = resolution
        self.transform = transform
        self.target_transform = target_transform

        # e.g. datasets/DAVIS/ImageSets/1080p/train.txt
        subset_file = os.path.join(self.root_dir, "ImageSets", self.resolution, f"{subset}.txt")
        with open(subset_file, "r") as f:
            sequences = [line.strip() for line in f if line.strip()]

        # Gather all (image_path, annotation_path) pairs
        self.samples = []
        for seq in sequences:
            seq_jpeg_dir = os.path.join(self.root_dir, "JPEGImages", self.resolution, seq)
            seq_anno_dir = os.path.join(self.root_dir, "Annotations", self.resolution, seq)

            if not os.path.isdir(seq_jpeg_dir):
                # If a listed sequence folder doesn't exist, skip or raise an error
                print(f"Warning: JPEG directory {seq_jpeg_dir} not found, skipping.")
                continue

            # Sort frames by filename to keep them in order
            frames = sorted(os.listdir(seq_jpeg_dir))
            for frame_name in frames:
                if not frame_name.endswith(".jpg"):
                    continue
                frame_id = os.path.splitext(frame_name)[0]  # e.g. frame_00000
                image_path = os.path.join(seq_jpeg_dir, frame_name)
                
                # Corresponding annotation
                anno_path = os.path.join(seq_anno_dir, frame_id + ".png")
                if not os.path.exists(anno_path):
                    # If annotation missing, skip or handle differently
                    print(f"Warning: annotation {anno_path} not found, skipping.")
                    continue

                self.samples.append((image_path, anno_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, anno_path = self.samples[idx]
        
        # Open image (RGB) and annotation (grayscale)
        image = Image.open(image_path).convert("RGB")
        annotation = Image.open(anno_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            annotation = self.target_transform(annotation)

        # Typically, annotations are loaded as single-channel masks.
        # If you do not have a special transform, you can convert it here:
        annotation = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(annotation.tobytes()))
            .view(annotation.size[1], annotation.size[0])
            .numpy()
        ).long()  # e.g. shape (H, W), dtype long

        return image, annotation
    
    from torch.utils.data import DataLoader

def get_davis_dataloader(root_dir="datasets/DAVIS",
                         subset="train",
                         resolution="480p",
                         transform=None,
                         target_transform=None,
                         batch_size=1,
                         shuffle=True,
                         num_workers=4):
    dataset = DAVISDataset(
        root_dir=root_dir,
        subset=subset,
        resolution=resolution,
        transform=transform,
        target_transform=target_transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader