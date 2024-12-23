# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 29

EPOCHS = 20
BATCH_SIZE = 2
SEED = 42

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


# Paths to dataset folders
SUNNY_IMAGES_PATH = '/net/ens/am4ip/datasets/project-dataset/sunny_images'
SUNNY_MASKS_PATH = '/net/ens/am4ip/datasets/project-dataset/sunny_sseg'
RAINY_IMAGES_PATH = '/net/ens/am4ip/datasets/project-dataset/rainy_images'
RAINY_MASKS_PATH = '/net/ens/am4ip/datasets/project-dataset/rainy_sseg'