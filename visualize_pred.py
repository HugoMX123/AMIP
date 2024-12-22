import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 29

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

def create_dataset(image_paths, mask_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# IoU Metric
def iou_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    intersection = tf.reduce_sum(tf.cast(y_pred * y_true, tf.float32))
    union = tf.reduce_sum(tf.cast(y_pred, tf.float32)) + tf.reduce_sum(tf.cast(y_true, tf.float32)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou

# Data Pipeline
def parse_image(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.numpy_function(func=rgb_to_class, inp=[mask], Tout=tf.uint8)

    # Explicitly set the shape for TensorFlow
    mask.set_shape([IMG_HEIGHT, IMG_WIDTH])
    mask = tf.one_hot(mask, NUM_CLASSES)
    mask.set_shape([IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES])

    return img, mask

# RGB to Class Mapping
def rgb_to_class(mask):
    mapping = {
        (0, 0, 0): 0, (111, 74, 0): 1, (81, 0, 81): 2, (128, 64, 128): 3, (244, 35, 232): 4,
        (250, 170, 160): 5, (230, 150, 140): 6, (70, 70, 70): 7, (102, 102, 156): 8, (190, 153, 153): 9,
        (180, 165, 180): 10, (150, 100, 100): 11, (150, 120, 90): 12, (153, 153, 153): 13, (250, 170, 30): 14,
        (220, 220, 0): 15, (107, 142, 35): 16, (152, 251, 152): 17, (70, 130, 180): 18, (220, 20, 60): 19,
        (255, 0, 0): 20, (0, 0, 142): 21, (0, 0, 70): 22, (0, 60, 100): 23, (0, 0, 90): 24,
        (0, 0, 110): 25, (0, 80, 100): 26, (0, 0, 230): 27, (119, 11, 32): 28
    }
    mask = mask.reshape(-1, 3)
    mask_class = np.zeros(mask.shape[0], dtype=np.uint8)
    for i, pixel in enumerate(mask):
        mask_class[i] = mapping.get(tuple(pixel), 0)
    return mask_class.reshape(IMG_HEIGHT, IMG_WIDTH)

# Paths to dataset folders
sunny_images_path = '/net/ens/am4ip/datasets/project-dataset/sunny_images'
sunny_masks_path = '/net/ens/am4ip/datasets/project-dataset/sunny_sseg'
rainy_images_path = '/net/ens/am4ip/datasets/project-dataset/rainy_images'
rainy_masks_path = '/net/ens/am4ip/datasets/project-dataset/rainy_sseg'

# Load file paths
sunny_image_files = [f for f in os.listdir(sunny_images_path) if f.endswith('.png')]
rainy_image_files = [f for f in os.listdir(rainy_images_path) if f.endswith('.png')]

sunny_image_paths = [os.path.join(sunny_images_path, f) for f in sunny_image_files]
sunny_mask_paths = [os.path.join(sunny_masks_path, f) for f in sunny_image_files]

rainy_image_paths = [os.path.join(rainy_images_path, f) for f in rainy_image_files]
rainy_mask_paths = [os.path.join(rainy_masks_path, f) for f in rainy_image_files]

image_paths = sunny_image_paths + rainy_image_paths
mask_paths = sunny_mask_paths + rainy_mask_paths

# Split datasets
train_img, val_test_img, train_mask, val_test_mask = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)
val_img, test_img, val_mask, test_mask = train_test_split(
    val_test_img, val_test_mask, test_size=0.5, random_state=42
)

batch_size = 8
train_dataset = create_dataset(train_img, train_mask, batch_size)
val_dataset = create_dataset(val_img, val_mask, batch_size)
test_dataset = create_dataset(test_img, test_mask, batch_size)



# Path to the saved model (adjust the epoch number as needed)
checkpoint_path = "checkpoints/epoch_17.h5"

# Load the model and pass the custom metric
model = load_model(checkpoint_path, custom_objects={'iou_metric': iou_metric})


visualize_predictions(model, test_dataset, num_samples=5)
