BATCH_SIZE = 16
METRIC = "MeanIoU"
SPECIALIZATION = "best_denoised"


# COMMON PARAMETERS
# saving path
SAVING_PATH = '/net/travail/hsosapavonca/AMIP/'

# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 29

# Model
MODEL_NAME = "U-Net"
LOSS_FUNCTION = "dice"

# Training hyperparameters
SEED = 42

EPOCHS = 30
LEARNING_RATE = 1e-3
#METRIC = "MeanIoU"

# Dataset split
DIVIDE_FIRST = True
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Data processings
DATA_AUGMENTATION = False
DATA_AUGMENTATION_LIST = []
DATA_DISCARDING_ACCORDING_TO_NOISE = True
DATA_NOISE_LAPLACIAN_THRESHOLD = 25

# Not implemented yet
DATA_DENOIZING = True


# Paths to dataset folders
SUNNY_IMAGES_PATH = '/net/ens/am4ip/datasets/project-dataset/sunny_images'
SUNNY_MASKS_PATH = '/net/ens/am4ip/datasets/project-dataset/sunny_sseg'
RAINY_IMAGES_PATH = '/net/ens/am4ip/datasets/project-dataset/rainy_images'
RAINY_MASKS_PATH = '/net/ens/am4ip/datasets/project-dataset/rainy_sseg'
