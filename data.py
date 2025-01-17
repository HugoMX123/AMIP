import cv2
import numpy as np
from config import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from denoizing import denoise_dataset

# New function to check noise using Laplacian variance
def is_noisy(img_path, threshold):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold

def filter_noise(image_paths, mask_paths):
    filtered_images = []
    filtered_masks = []
    discarded_count = 0
    for img, mask in zip(image_paths, mask_paths):
        if not is_noisy(img, threshold=DATA_NOISE_LAPLACIAN_THRESHOLD):
            filtered_images.append(img)
            filtered_masks.append(mask)
        else:
            discarded_count += 1
    print(f"Discarded {discarded_count} out of {len(image_paths)} images due to noise.")
    image_paths = filtered_images
    mask_paths = filtered_masks
    return image_paths, mask_paths

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

# Augment function for horizontal flipping (you can expand this for more augmentations)
def augment_image(img, mask):
    if "horizontal_flip" in DATA_AUGMENTATION_LIST:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    
    return img, mask

def create_dataset(image_paths, mask_paths, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation if the flag is True
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def load_datasets(FILTER_NOISY=DATA_DISCARDING_ACCORDING_TO_NOISE, noise_threshold=DATA_NOISE_LAPLACIAN_THRESHOLD, AUGMENT_DATA=DATA_AUGMENTATION, DATA_DENOIZING=DATA_DENOIZING):

    # Load file paths
    sunny_image_files = [f for f in os.listdir(SUNNY_IMAGES_PATH) if f.endswith('.png')]
    rainy_image_files = [f for f in os.listdir(RAINY_IMAGES_PATH) if f.endswith('.png')]

    sunny_image_paths = [os.path.join(SUNNY_IMAGES_PATH, f) for f in sunny_image_files]
    sunny_mask_paths = [os.path.join(SUNNY_MASKS_PATH, f) for f in sunny_image_files]

    rainy_image_paths = [os.path.join(RAINY_IMAGES_PATH, f) for f in rainy_image_files]
    rainy_mask_paths = [os.path.join(RAINY_MASKS_PATH, f) for f in rainy_image_files]

    if DIVIDE_FIRST:
        print("FIRST DIVIDING SUNNY AND RAINY")
        if FILTER_NOISY:
            rainy_image_paths, rainy_mask_paths = filter_noise(rainy_image_paths, rainy_mask_paths)
        
        # Split sunny datasets
        sunny_train_img, sunny_val_test_img, sunny_train_mask, sunny_val_test_mask = train_test_split(
            sunny_image_paths, sunny_mask_paths, test_size=(1-TRAIN_SPLIT), random_state=SEED)
        
        sunny_val_img, sunny_test_img, sunny_val_mask, sunny_test_mask = train_test_split(
            sunny_val_test_img, sunny_val_test_mask, test_size=(TEST_SPLIT/(TEST_SPLIT + VAL_SPLIT)), random_state=SEED)
        
        # Split rainy datasets
        rainy_train_img, rainy_val_test_img, rainy_train_mask, rainy_val_test_mask = train_test_split(
            rainy_image_paths, rainy_mask_paths, test_size=(1-TRAIN_SPLIT), random_state=SEED)
        
        rainy_val_img, rainy_test_img, rainy_val_mask, rainy_test_mask = train_test_split(
            rainy_val_test_img, rainy_val_test_mask, test_size=(TEST_SPLIT/(TEST_SPLIT + VAL_SPLIT)), random_state=SEED)
        
        train_img = sunny_train_img + rainy_train_img
        train_mask = sunny_train_mask + rainy_train_mask
        val_img = sunny_val_img + rainy_val_img
        val_mask = sunny_val_mask + rainy_val_mask
        test_img = sunny_test_img + rainy_test_img
        test_mask = sunny_test_mask + rainy_test_mask

        # Create datasets, apply augmentation if enabled
        train_dataset = create_dataset(train_img, train_mask, BATCH_SIZE, augment=AUGMENT_DATA)
        val_dataset = create_dataset(val_img, val_mask, BATCH_SIZE)
        test_dataset = create_dataset(test_img, test_mask, BATCH_SIZE)

    else:
        image_paths = sunny_image_paths + rainy_image_paths
        mask_paths = sunny_mask_paths + rainy_mask_paths

        # Filter noisy images if enabled
        if FILTER_NOISY:
            image_paths, mask_paths = filter_noise(image_paths, mask_paths)

        # Split datasets
        train_img, val_test_img, train_mask, val_test_mask = train_test_split(
            image_paths, mask_paths, test_size=(1-TRAIN_SPLIT), random_state=SEED)
        
        val_img, test_img, val_mask, test_mask = train_test_split(
            val_test_img, val_test_mask, test_size=(TEST_SPLIT/(TEST_SPLIT + VAL_SPLIT)), random_state=SEED)

        # Create datasets, apply augmentation if enabled
        train_dataset = create_dataset(train_img, train_mask, BATCH_SIZE, augment=AUGMENT_DATA)
        val_dataset = create_dataset(val_img, val_mask, BATCH_SIZE)
        test_dataset = create_dataset(test_img, test_mask, BATCH_SIZE)

    if DATA_DENOIZING:
        train_dataset = denoise_dataset(train_dataset)


    return train_dataset, val_dataset, test_dataset


