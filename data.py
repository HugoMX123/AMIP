import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torch.utils.data import random_split

def get_dataloaders(image_dirs, mask_dirs, batch_size, image_size, split_ratios=(0.8, 0.1, 0.1)):
    """
    Returns DataLoaders for training, validation, and testing.
    Args:
        image_dirs (list): List of image directory paths.
        mask_dirs (list): List of corresponding mask directory paths.
        batch_size (int): Batch size for DataLoader.
        image_size (tuple): Image resizing dimensions.
        split_ratios (tuple): Ratios for train, val, and test splits.
    """
    dataset = InstanceSegmentationDataset(image_dirs, mask_dirs, image_size=image_size)
    total_len = len(dataset)
    
    train_len = int(total_len * split_ratios[0])
    val_len = int(total_len * split_ratios[1])
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=16)
    print("Trainer loader finished!")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=16)
    print("Validation loader finished!")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=16)
    print("Test loader finished!")
    print("Data loaders finished")
    return train_loader, val_loader, test_loader


class InstanceSegmentationDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None, image_size=(512, 512), unlabeled_id=255):
        """
        Initialize dataset with multiple image and mask directories.
        Args:
            image_dirs (list): List of image directory paths.
            mask_dirs (list): List of corresponding mask directory paths.
            transform (callable, optional): Optional transform to be applied.
            image_size (tuple): Image resizing dimensions.
            unlabeled_id (int): Pixel value for unlabeled regions.
        """
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.image_size = image_size
        self.unlabeled_id = unlabeled_id
        
        # Aggregate image paths across all directories
        self.images = []
        self.masks = []
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            self.images.extend([os.path.join(image_dir, f) for f in image_files])
            self.masks.extend([os.path.join(mask_dir, f) for f in image_files])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Resize and convert
        image = F.resize(image, self.image_size)
        mask = F.resize(mask, self.image_size, interpolation=Image.NEAREST)
        mask = np.array(mask)

        # Map unlabeled to background (0)
        mask[mask == self.unlabeled_id] = 0

        # Get unique object IDs (excluding background)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        # Create binary masks and bounding boxes
        masks = mask == obj_ids[:, None, None]
        boxes = []
        valid_masks = []
        valid_labels = []
        for i, obj_id in enumerate(obj_ids):
            pos = np.where(masks[i])
            if len(pos[0]) > 0 and len(pos[1]) > 0:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                
                # Ensure the box is valid
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    valid_masks.append(masks[i])
                    valid_labels.append(obj_id)

        # Ensure at least one valid object
        if len(boxes) == 0:
            raise ValueError(f"No valid bounding boxes found for image {image_path}")

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(valid_labels, dtype=torch.int64)
        masks = torch.tensor(np.array(valid_masks), dtype=torch.uint8)
        
        return F.to_tensor(image), {"boxes": boxes, "labels": labels, "masks": masks}



