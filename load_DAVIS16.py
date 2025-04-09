import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DAVISDataset(Dataset):
    """
    A Dataset for DAVIS-16 single-object segmentation with target and reference images,
    organized by sequence title (for example, "bear" or "bmx-bumps").
    
    Each line in the train.txt file is assumed to contain:
      <image_path> <annotation_path>
      
    The image_path is expected to follow a folder structure such as:
      /JPEGImages/480p/{title}/{frame}.jpg
      
    The initialization groups samples by title. An optional parameter "selected_titles"
    can be provided to restrict the dataset to only those titles.
    
    For each sample, the dataset returns:
      - target_image: the image for which segmentation is performed.
      - reference_images: a list of all other images in the same title group.
      - target_annotation: the corresponding annotation for the target image.
    """
    def __init__(self,
                 root_dir="datasets/DAVIS",
                 subset="train",
                 resolution="480p",
                 transform=None,
                 target_transform=None,
                 selected_titles=None):
        super().__init__()
        self.root_dir = root_dir
        self.resolution = resolution
        self.transform = transform
        self.target_transform = target_transform
        # Optional: if provided, only include samples from these titles.
        self.selected_titles = selected_titles

        # We assume the training file is located at:
        # <root_dir>/ImageSets/<resolution>/<subset>.txt
        # and each line contains: 
        #   /JPEGImages/<resolution>/<title>/<frame>.jpg /Annotations/<resolution>/<title>/<frame>.png
        train_file = os.path.join(root_dir, "ImageSets", resolution, f"{subset}.txt")
        self.samples_by_title = {}  # Will map title -> list of (image_path, annotation_path)
        
        with open(train_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                # Skip lines that don't have exactly two paths.
                continue
            rel_img_path, rel_anno_path = parts

            # Normalize path in case of leading '/'
            norm_img_path = os.path.normpath(rel_img_path)
            split_parts = norm_img_path.split(os.sep)
            # We assume the path structure is something like:
            # ['JPEGImages', '480p', 'bear', '00074.jpg']
            # So the title is the element right after the resolution.
            try:
                # Find the index where the resolution string occurs.
                res_index = split_parts.index(resolution)
                # The next element should be the title.
                title = split_parts[res_index + 1]
            except (ValueError, IndexError):
                title = "unknown"
            
            # If selected_titles is provided, skip samples not in that list.
            if self.selected_titles is not None and title not in self.selected_titles:
                continue

            # Build full paths by joining with the root_dir and stripping any leading slashes.
            full_img_path = os.path.join(root_dir, rel_img_path.lstrip("/"))
            full_anno_path = os.path.join(root_dir, rel_anno_path.lstrip("/"))
            
            # Insert the sample into the dictionary for that title.
            if title not in self.samples_by_title:
                self.samples_by_title[title] = []
            self.samples_by_title[title].append((full_img_path, full_anno_path))
        
        # Sort samples within each title by file name to maintain a consistent order.
        for title in self.samples_by_title:
            self.samples_by_title[title] = sorted(
                self.samples_by_title[title],
                key=lambda x: os.path.basename(x[0])
            )
        
        # Create a flat index list where each entry is (title, index_within_title)
        self.flat_indices = []
        for title, sample_list in self.samples_by_title.items():
            for idx in range(len(sample_list)):
                self.flat_indices.append((title, idx))

    def __len__(self):
        return len(self.flat_indices)
    
    def __getitem__(self, index):
        # Retrieve the title and the index within that title group.
        title, idx = self.flat_indices[index]
        target_img_path, target_anno_path = self.samples_by_title[title][idx]
        
        # Load the target image and annotation.
        target_image = Image.open(target_img_path).convert("RGB")
        target_anno = Image.open(target_anno_path)
        
        if self.transform is not None:
            target_image = self.transform(target_image)
        if self.target_transform is not None:
            target_anno = self.target_transform(target_anno)
        else:
            # By default, convert the annotation to a tensor.
            target_anno = torch.from_numpy(
                np.array(target_anno, dtype=np.uint8)
            ).long()

        # Retrieve reference images (all images from the same title except the target).
        reference_images = []
        for i, (img_path, _) in enumerate(self.samples_by_title[title]):
            if i == idx:
                continue  # Skip the target image.
            ref_img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                ref_img = self.transform(ref_img)
            reference_images.append(ref_img)
        
        return target_image, reference_images, target_anno

def get_davis_dataloader(root_dir="datasets/DAVIS",
                         subset="train",
                         resolution="480p",
                         transform=None,
                         target_transform=None,
                         selected_titles=None,
                         batch_size=1,
                         shuffle=True,
                         num_workers=4):
    dataset = DAVISDataset(
        root_dir=root_dir,
        subset=subset,
        resolution=resolution,
        transform=transform,
        target_transform=target_transform,
        selected_titles=selected_titles
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader