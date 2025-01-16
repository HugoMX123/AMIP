import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from config import *

def get_metric():
    if METRIC == "MeanIoU":
        chosen_metric = tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)
        monitor_metric = "val_mean_io_u"
    elif METRIC == "DICE":
        chosen_metric = dice_metric
        monitor_metric = "val_dice_metric"
    else:
        raise ValueError(f"Unknown metric: {METRIC}")
    
    return chosen_metric, monitor_metric

# IoU Metric
def iou_metric(y_true, y_pred):
    # Convert predictions and labels to class indices
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)

    # Compute IoU for each class
    iou_list = []
    for i in range(NUM_CLASSES):
        y_true_class = tf.cast(y_true == i, tf.float32)
        y_pred_class = tf.cast(y_pred == i, tf.float32)
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)  # Avoid divide by zero
        iou_list.append(iou)

    # Compute mean IoU
    mean_iou = tf.reduce_mean(tf.stack(iou_list))
    return mean_iou

def dice_metric():
    pass

def save_learning_curve(history, monitor_metric):
    # Plot training and validation chosen metric (e.g., accuracy)
    plt.plot(history.history[monitor_metric[4:]], label=f'Training {monitor_metric.capitalize()}')
    plt.plot(history.history[monitor_metric], label=f'Validation {monitor_metric.capitalize()}')
    plt.title(f'{monitor_metric.capitalize()} Curves')
    plt.xlabel('Epochs')
    plt.ylabel(monitor_metric.capitalize())
    plt.legend()

    # Save the plots
    plot_file_path = SAVING_PATH + 'learning_curves/' + MODEL_NAME + SPECIALIZATION + '_learning_curves.png'
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')

def save_results(history, test_acc, traning_time, add_spec = "", file_name = "results.xlsx"):

    if METRIC  == "MeanIoU":
        val_acc_history = history.history['val_mean_io_u']  # Validation accuracy per epoch
    
    val_loss_history = history.history['val_loss']    # Validation loss per epoch

    # Get the best validation accuracy and the corresponding loss
    valid_acc = max(val_acc_history)          # Best validation accuracy
    best_epoch = val_acc_history.index(valid_acc) + 1  # Epoch with best accuracy
    valid_loss = val_loss_history[best_epoch - 1] 

    attributes = ["Model name", "Specialization", "Loss function", "Metric", "Img channels", "Img height", "Img width", 
                  "Learning rate", "Batch size", "Train split", "Val split", "Test split", "Data augmentations", 
                  "Data discarding according to noise", "DATA_NOISE_LAPLACIAN_THRESHOLD", "DATA_DENOIZING", "All epoch", 
                  "Best epoch", "Best validation loss", "Best validation accuracy", "Test accuracy", "traning time (s)"]
    new_row = {
        "Img channels": IMG_CHANNELS,
        "Img height": IMG_HEIGHT,
        "Img width": IMG_WIDTH,
        "Model name": MODEL_NAME,
        "Specialization": SPECIALIZATION + add_spec,
        "Loss function": LOSS_FUNCTION,
        "Learning rate": LEARNING_RATE,
        "Batch size": BATCH_SIZE,
        "Metric": METRIC,
        "Train split": TRAIN_SPLIT,
        "Val split": VAL_SPLIT,
        "Test split": TEST_SPLIT,
        "Data augmentations": DATA_AUGMENTATION_LIST,
        "Data discarding according to noise": DATA_DISCARDING_ACCORDING_TO_NOISE,
        "DATA_NOISE_LAPLACIAN_THRESHOLD": DATA_NOISE_LAPLACIAN_THRESHOLD,
        "DATA_DENOIZING": DATA_DENOIZING,
        "All epoch": EPOCHS,  
        "Best epoch": best_epoch,
        "Best validation loss": valid_loss,
        "Best validation accuracy": valid_acc,
        "Test accuracy": test_acc,
        "traning time (s)": traning_time
    }

    # Check if the file exists
    if os.path.exists(file_name):
        # Read the existing table
        existing_df = pd.read_excel(file_name)
    else:
        # Create a new table if it doesn't exist
        existing_df = pd.DataFrame(columns=attributes)

    # Add the new row to the DataFrame
    existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame to the Excel file
    existing_df.to_excel(file_name, index=False)

    print(f"Table updated and saved to {file_name}")