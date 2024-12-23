import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from config import *
from model import iou_metric
from data_pipeline import create_dataset


# Visualize predictions
def visualize_predictions(model, dataset, num_samples=3):
    for images, masks in dataset.take(1):  # Take one batch
        predictions = model.predict(images)
        predictions = tf.argmax(predictions, axis=-1)  # Convert to class indices
        masks = tf.argmax(masks, axis=-1)  # Ground truth as class indices

        for i in range(num_samples):
            plt.figure(figsize=(12, 4))

            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(images[i].numpy())
            plt.title("Input Image")
            plt.axis("off")

            # Ground Truth Mask
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i].numpy(), cmap="nipy_spectral")
            plt.title("Ground Truth")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(predictions[i].numpy(), cmap="nipy_spectral")
            plt.title("Prediction")
            plt.axis("off")

            plt.show()






# Load file paths
sunny_image_files = [f for f in os.listdir(SUNNY_IMAGES_PATH) if f.endswith('.png')]
rainy_image_files = [f for f in os.listdir(RAINY_IMAGES_PATH) if f.endswith('.png')]

sunny_image_paths = [os.path.join(SUNNY_IMAGES_PATH, f) for f in sunny_image_files]
sunny_mask_paths = [os.path.join(SUNNY_MASKS_PATH, f) for f in sunny_image_files]

rainy_image_paths = [os.path.join(RAINY_IMAGES_PATH, f) for f in rainy_image_files]
rainy_mask_paths = [os.path.join(RAINY_MASKS_PATH, f) for f in rainy_image_files]

image_paths = sunny_image_paths + rainy_image_paths
mask_paths = sunny_mask_paths + rainy_mask_paths

# Split datasets
train_img, val_test_img, train_mask, val_test_mask = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=SEED
)
val_img, test_img, val_mask, test_mask = train_test_split(
    val_test_img, val_test_mask, test_size=0.5, random_state=SEED
)


train_dataset = create_dataset(train_img, train_mask, BATCH_SIZE)
val_dataset = create_dataset(val_img, val_mask, BATCH_SIZE)
test_dataset = create_dataset(test_img, test_mask, BATCH_SIZE)



# Path to the saved model (adjust the epoch number as needed)
checkpoint_path = "checkpoints/epoch_17.h5"

# Load the model and pass the custom metric
model = load_model(checkpoint_path, custom_objects={'iou_metric': iou_metric})


visualize_predictions(model, test_dataset, num_samples=5)
