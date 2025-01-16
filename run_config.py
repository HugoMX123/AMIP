import os
import subprocess

# Define the parameter combinations
configurations = [
    {"BATCH_SIZE": 4, "SPECIALIZATION": "_2_DIVIDE_FIRST"},
    {"BATCH_SIZE": 8, "SPECIALIZATION": "_3_DIVIDE_FIRST"},
    {"BATCH_SIZE": 16, "SPECIALIZATION": "_4_DIVIDE_FIRST"},
    {"BATCH_SIZE": 32, "SPECIALIZATION": "_5_DIVIDE_FIRST"},
]

# Path to parameters.py and main.py
parameters_file = "config.py"
main_script = "main.py"

# Backup the original parameters file
if not os.path.exists(f"{parameters_file}.backup"):
    os.rename(parameters_file, f"{parameters_file}.backup")

# Function to update parameters.py
def update_parameters(config):
    with open(parameters_file, "w") as file:
        file.write(f"""BATCH_SIZE = {config['BATCH_SIZE']}\n""")
        file.write(f"""SPECIALIZATION = "{config['SPECIALIZATION']}"\n\n""")
        file.write("""
# COMMON PARAMETERS
# saving path
SAVING_PATH = '/net/travail/mvajay/advanced_project/'

# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 29

# Model
MODEL_NAME = "U-Net"
LOSS_FUNCTION = "categorical_crossentropy"

# Training hyperparameters
SEED = 42

EPOCHS = 50
LEARNING_RATE = 1e-3
METRIC = "MeanIoU"

# Dataset split
DIVIDE_FIRST = True
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Data processings
DATA_AUGMENTATION = False
DATA_AUGMENTATION_LIST = []
DATA_DISCARDING_ACCORDING_TO_NOISE = False
DATA_NOISE_LAPLACIAN_THRESHOLD = 25

# Not implemented yet
DATA_DENOIZING = False


# Paths to dataset folders
SUNNY_IMAGES_PATH = '/net/ens/am4ip/datasets/project-dataset/sunny_images'
SUNNY_MASKS_PATH = '/net/ens/am4ip/datasets/project-dataset/sunny_sseg'
RAINY_IMAGES_PATH = '/net/ens/am4ip/datasets/project-dataset/rainy_images'
RAINY_MASKS_PATH = '/net/ens/am4ip/datasets/project-dataset/rainy_sseg'
""")

# Iterate over each configuration
for config in configurations:
    print(f"Running configuration: {config}")
    # Update the parameters.py file
    update_parameters(config)
    # Execute the main.py script
    try:
        subprocess.run(["python", main_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running configuration {config}: {e}")

# Restore the original parameters.py file
os.rename(f"{parameters_file}.backup", parameters_file)
