import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split

# Paths to dataset folders
sunny_images_path = '/net/ens/am4ip/datasets/project-dataset/sunny_images'
sunny_masks_path = '/net/ens/am4ip/datasets/project-dataset/sunny_sseg'
rainy_images_path = '/net/ens/am4ip/datasets/project-dataset/rainy_images'
rainy_masks_path = '/net/ens/am4ip/datasets/project-dataset/rainy_sseg'

# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 29

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Disable XLA and OneDNN optimizations globally
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check available devices
print("Available devices:", tf.config.list_physical_devices())

# RGB to Class Mapping
def rgb_to_class(mask):
    mapping = {
        (0, 0, 0): 0, 
        (111, 74, 0): 1, 
        (81, 0, 81): 2, 
        (128, 64, 128): 3, 
        (244, 35, 232): 4,
        (250, 170, 160): 5, 
        (230, 150, 140): 6, 
        (70, 70, 70): 7, 
        (102, 102, 156): 8, 
        (190, 153, 153): 9,
        (180, 165, 180): 10, 
        (150, 100, 100): 11, 
        (150, 120, 90): 12, 
        (153, 153, 153): 13, 
        (250, 170, 30): 14,
        (220, 220, 0): 15, 
        (107, 142, 35): 16, 
        (152, 251, 152): 17, 
        (70, 130, 180): 18, 
        (220, 20, 60): 19,
        (255, 0, 0): 20, 
        (0, 0, 142): 21, 
        (0, 0, 70): 22, 
        (0, 60, 100): 23, 
        (0, 0, 90): 24,
        (0, 0, 110): 25, 
        (0, 80, 100): 26, 
        (0, 0, 230): 27, 
        (119, 11, 32): 28
    }
    mask = mask.reshape(-1, 3)
    mask_class = np.zeros(mask.shape[0], dtype=np.uint8)
    for i, pixel in enumerate(mask):
        mask_class[i] = mapping.get(tuple(pixel), 0)
    return mask_class.reshape(IMG_HEIGHT, IMG_WIDTH)

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

def create_dataset(image_paths, mask_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

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

batch_size = 2
train_dataset = create_dataset(train_img, train_mask, batch_size)
val_dataset = create_dataset(val_img, val_mask, batch_size)
test_dataset = create_dataset(test_img, test_mask, batch_size)

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


# U-Net Model
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    u1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[iou_metric])
    return model

model = unet_model()

# Save model after each epoch
checkpoint_path = "checkpoints/epoch_{epoch:02d}.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_best_only=False)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=30, callbacks=[checkpoint_callback])
