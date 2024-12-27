# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 29

EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
SEED = 42

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

DATA_AUGMENTATION = False
DATA_DISCARDING_ACCORDING_TO_NOISE = False
DATA_NOISE_LAPLACIAN_THRESHOLD = 25

# Not implemented yet
DATA_DENOIZING = False


# Paths to dataset folders
SUNNY_IMAGES_PATH = '/net/ens/am4ip/datasets/project-dataset/sunny_images'
SUNNY_MASKS_PATH = '/net/ens/am4ip/datasets/project-dataset/sunny_sseg'
RAINY_IMAGES_PATH = '/net/ens/am4ip/datasets/project-dataset/rainy_images'
RAINY_MASKS_PATH = '/net/ens/am4ip/datasets/project-dataset/rainy_sseg'