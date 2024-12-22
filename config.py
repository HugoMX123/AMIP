# config.py

import torch

# Paths
#CSV_PATH = '/net/travail/hsosapavonca/AMIP/segmentation_colors.csv'
SUNNY_IMAGE_DIR = "/net/ens/am4ip/datasets/project-dataset/sunny_images"
SUNNY_MASK_DIR = "/net/ens/am4ip/datasets/project-dataset/sunny_sseg"
RAINY_IMAGE_DIR = "/net/ens/am4ip/datasets/project-dataset/rainy_images"
RAINY_MASK_DIR = "/net/ens/am4ip/datasets/project-dataset/rainy_sseg"

IMAGES_DIR = [SUNNY_IMAGE_DIR, RAINY_IMAGE_DIR]
MASKS_DIR = [SUNNY_MASK_DIR, RAINY_MASK_DIR]

MODEL_SAVE_PATH = "/net/travail/hsosapavonca/AMIP/model/model.pth"  # Save path for the model

# Hyperparameters
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
#NUM_CLASSES = 33
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

