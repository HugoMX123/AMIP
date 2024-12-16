import cv2
import os
import numpy as np

def overlay_images(image, mask, alpha=0.5):
    """
    Overlay the mask on the image with a given transparency.
    
    :param image: Original image (BGR)
    :param mask: Segmentation mask (BGR)
    :param alpha: Transparency factor for the mask.
    :return: Overlaid image
    """
    return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

def resize_image(image, width=None, height=None):
    """
    Resize the image to the specified width and height while maintaining aspect ratio.

    :param image: Input image
    :param width: Desired width (optional)
    :param height: Desired height (optional)
    :return: Resized image
    """
    if width is None and height is None:
        return image

    h, w = image.shape[:2]
    if width is None:
        scale = height / h
        new_size = (int(w * scale), height)
    elif height is None:
        scale = width / w
        new_size = (width, int(h * scale))
    else:
        new_size = (width, height)

    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

def visualize_dataset(image_folder, mask_folder, width=None, height=None):
    """
    Visualize images and segmentation masks overlapped like a video with navigation controls.

    :param image_folder: Path to the folder containing images
    :param mask_folder: Path to the folder containing segmentation masks
    :param width: Desired width for resizing images (optional)
    :param height: Desired height for resizing images (optional)
    """
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    # Ensure image and mask filenames match
    common_files = [f for f in image_files if f in mask_files]
    if not common_files:
        print("No matching files found between image and mask folders.")
        return

    idx = 0
    playing = True

    while True:
        # Load the current image and mask
        image_path = os.path.join(image_folder, common_files[idx])
        mask_path = os.path.join(mask_folder, common_files[idx])

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        if image is None or mask is None:
            print(f"Failed to load image or mask: {common_files[idx]}")
            break

        # Resize images if dimensions are provided
        if width or height:
            image = resize_image(image, width, height)
            mask = resize_image(mask, width, height)

        # Overlay the mask on the image
        overlaid = overlay_images(image, mask)

        # Put the name of the file on the image
        cv2.putText(overlaid, common_files[idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the overlaid image
        cv2.imshow("Image with Mask", overlaid)

        if playing:
            # Automatically move to the next frame after a delay
            key = cv2.waitKey(33)  # 100 ms delay (10 FPS)
            idx = (idx + 1) % len(common_files)
        else:
            key = cv2.waitKey(0)  # Wait indefinitely for a key press

        if key == 27:  # ESC key to exit
            break
        elif key == ord('a'):  # 'A' key to go back
            idx = (idx - 1) % len(common_files)
        elif key == ord('d'):  # 'D' key to go forward
            idx = (idx + 1) % len(common_files)
        elif key == ord('p'):  # 'P' key to pause/play
            playing = not playing

    cv2.destroyAllWindows()

# Usage example
image_folder = "/net/ens/am4ip/datasets/project-dataset/rainy_images"
mask_folder = "/net/ens/am4ip/datasets/project-dataset/rainy_sseg"

dataset = 1 # 0 for Sunny - 1 for Rainy

if dataset == 0:
    image_folder = "/net/ens/am4ip/datasets/project-dataset/sunny_images"
    mask_folder = "/net/ens/am4ip/datasets/project-dataset/sunny_sseg"

else:
    image_folder = "/net/ens/am4ip/datasets/project-dataset/rainy_images"
    mask_folder = "/net/ens/am4ip/datasets/project-dataset/rainy_sseg"


visualize_dataset(image_folder, mask_folder, width=640*2, height=480*2)
