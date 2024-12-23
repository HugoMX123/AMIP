import numpy as np
from config import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

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

def load_datasets():
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Disable XLA and OneDNN optimizations globally
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Check available devices
    print("Available devices:", tf.config.list_physical_devices())

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
        image_paths, mask_paths, test_size=(1-TRAIN_SPLIT), random_state=SEED
    )
    val_img, test_img, val_mask, test_mask = train_test_split(
        val_test_img, val_test_mask, test_size=(TEST_SPLIT/(TEST_SPLIT + VAL_SPLIT)), random_state=SEED
    )

    train_dataset = create_dataset(train_img, train_mask, BATCH_SIZE)
    val_dataset = create_dataset(val_img, val_mask, BATCH_SIZE)
    test_dataset = create_dataset(test_img, test_mask, BATCH_SIZE)

    return train_dataset, val_dataset, test_dataset 