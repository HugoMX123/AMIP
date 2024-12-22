import torch
from torchvision.ops import box_iou
from config import *
from data import get_dataloaders
from model import get_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load color mapping from segmentation_colors.csv

colors_df = pd.read_csv(CSV_PATH)
color_dict = {row['ID']: tuple(map(int, row['Segmentation Color'].strip("()").split(','))) for _, row in colors_df.iterrows()}

def apply_color_mask(mask, color_dict):
    """
    Convert a mask into an RGB image using the color mapping.
    Args:
        mask: 2D numpy array of class IDs.
        color_dict: Dictionary mapping class IDs to RGB colors.
    Returns:
        RGB mask image.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_dict.items():
        color_mask[mask == class_id] = color

    return color_mask

from PIL import Image

def visualize_predictions(image, pred_mask, gt_mask_path, color_dict):
    """
    Visualize the input image, predicted mask, and ground truth mask.
    Args:
        image: Tensor image (C, H, W).
        pred_mask: Predicted mask (H, W).
        gt_mask_path: Path to the ground truth mask image (colored jpg).
        color_dict: Color mapping dictionary.
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    pred_colored = apply_color_mask(pred_mask, color_dict)
    gt_mask = np.array(Image.open(gt_mask_path))  # Directly load the GT mask as an RGB image

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[1].imshow(pred_colored)
    axes[1].set_title("Predicted Mask")
    axes[2].imshow(gt_mask)
    axes[2].set_title("Ground Truth Mask")

    for ax in axes:
        ax.axis("off")
    plt.show()

def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)

    print("Starting evaluation and visualization...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            batch_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)

            # Visualize first 3 predictions
            for i in range(min(3, len(images))):
                pred_masks = outputs[i]['masks'].squeeze(1).cpu().numpy().argmax(0)  # Convert predictions to class IDs
                
                # Get corresponding ground truth mask path
                gt_mask_path = batch_targets[i]['path']  # You may need to pass paths in your dataset
                visualize_predictions(images[i], pred_masks, gt_mask_path, color_dict)
            break  # Visualize only one batch of images

# Load test data
_, _, test_loader = get_dataloaders(IMAGES_DIR, MASKS_DIR, BATCH_SIZE, IMAGE_SIZE)

# Load model
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load("/net/travail/hsosapavonca/AMIP/model/model_epoch_5.pth"))

# Evaluate the model with visualization
evaluate_model(model, test_loader, DEVICE)


