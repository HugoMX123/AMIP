import matplotlib.pyplot as plt
import tensorflow as tf
from config import *
from model import iou_metric
from data import load_datasets
from model import unet_model


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



train_dataset, val_dataset, test_dataset = load_datasets()
print("Train dataset size:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Validation dataset size:", tf.data.experimental.cardinality(val_dataset).numpy())
print("Test dataset size:", tf.data.experimental.cardinality(test_dataset).numpy())



# Path to the saved model (adjust the epoch number as needed)
checkpoint_path = "checkpoints/epoch_30.h5"

# Initialize the model architecture
model = unet_model()

# Load the weights from the checkpoint
model.load_weights(checkpoint_path)


visualize_predictions(model, test_dataset, num_samples=10)
