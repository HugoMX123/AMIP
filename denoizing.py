import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os



def fourier_denoising_color(image, radius_factor=0.2):
    """
    Apply Fourier-based denoising to a color image by processing each channel separately.
    Args:
        image: Input color image as a NumPy array (H, W, C).
        radius_factor: Fraction of the image dimensions to use as the radius of the low-pass filter.
    Returns:
        Denoised color image as a NumPy array (H, W, C).
    """
    # Function to process a single channel
    def process_channel(channel):
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        radius = int(min(rows, cols) * radius_factor)

        # Create a circular mask
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

        # Apply the mask
        fshift_filtered = fshift * mask

        # Inverse Fourier Transform
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize the result
        img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
        return (img_back * 255).astype(np.uint8)

    # Split the image into channels
    channels = cv2.split(image)

    # Process each channel
    denoised_channels = [process_channel(ch) for ch in channels]

    # Merge the channels back into a color image
    denoised_image = cv2.merge(denoised_channels)
    return denoised_image



# Wrapper to process a TensorFlow dataset
def denoise_dataset(dataset):
    """
    Apply Fourier denoising to all images in a TensorFlow dataset.
    Args:
        dataset: A TensorFlow dataset of (image, mask) pairs.

    Returns:
        A new TensorFlow dataset with denoised images.
    """
    def denoise_map_fn(image, mask):
        # Use tf.py_function to apply NumPy-based denoising
        denoised_image = tf.py_function(func=fourier_denoising_color, inp=[image], Tout=tf.uint8)
        denoised_image = tf.ensure_shape(denoised_image, image.shape)
        return denoised_image, mask

    return dataset.map(denoise_map_fn, num_parallel_calls=tf.data.AUTOTUNE)







