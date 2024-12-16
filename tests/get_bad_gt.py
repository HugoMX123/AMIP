import os
import cv2
import numpy as np
import torch

# Function to compute inter-frame differences using GPU
def compute_inter_frame_difference(image1_path, image2_path, gt1_path, gt2_path):
    # Load the input images and GT masks
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    gt1 = cv2.imread(gt1_path, cv2.IMREAD_GRAYSCALE)
    gt2 = cv2.imread(gt2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None or gt1 is None or gt2 is None:
        raise ValueError(f"Failed to load one of the images or GTs: {image1_path}, {image2_path}, {gt1_path}, {gt2_path}")

    # Ensure the images and GTs have the same dimensions
    if image1.shape != image2.shape or gt1.shape != gt2.shape:
        raise ValueError(f"Dimension mismatch between frames or GTs")

    # Convert to PyTorch tensors and move to GPU
    image1_tensor = torch.tensor(image1, dtype=torch.float32).cuda()
    image2_tensor = torch.tensor(image2, dtype=torch.float32).cuda()
    gt1_tensor = torch.tensor(gt1, dtype=torch.float32).cuda()
    gt2_tensor = torch.tensor(gt2, dtype=torch.float32).cuda()

    # Compute differences
    image_diff = torch.mean(torch.abs(image1_tensor - image2_tensor))
    gt_diff = torch.mean(torch.abs(gt1_tensor - gt2_tensor))

    return image_diff.item(), gt_diff.item()

# Function to evaluate inter-frame consistency
def evaluate_inter_frame_consistency(image_folder, gt_folder, diff_threshold=0.1):
    # Get list of image and GT files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) != len(gt_files):
        raise ValueError("Mismatch in the number of images and ground truth files")

    results = []
    total_files = len(image_files)

    # Process consecutive file pairs
    for idx in range(total_files - 1):
        image1_path = os.path.join(image_folder, image_files[idx])
        image2_path = os.path.join(image_folder, image_files[idx + 1])
        gt1_path = os.path.join(gt_folder, gt_files[idx])
        gt2_path = os.path.join(gt_folder, gt_files[idx + 1])

        try:
            image_diff, gt_diff = compute_inter_frame_difference(image1_path, image2_path, gt1_path, gt2_path)
            is_inconsistent = gt_diff > diff_threshold and image_diff < diff_threshold
            results.append({
                'frame1': image_files[idx],
                'frame2': image_files[idx + 1],
                'image_diff': image_diff,
                'gt_diff': gt_diff,
                'is_inconsistent': is_inconsistent
            })
        except Exception as e:
            print(f"Error processing {image_files[idx]} and {image_files[idx + 1]}: {e}")

        # Print progress
        print(f"Processed {idx + 1}/{total_files - 1} frame pairs ({(idx + 1) / (total_files - 1) * 100:.2f}% done)")

    return results

# Main function
if __name__ == "__main__":
    # Paths to the folders containing images and GTs
    image_folder = "/net/ens/am4ip/datasets/project-dataset/rainy_images"
    gt_folder = "/net/ens/am4ip/datasets/project-dataset/rainy_sseg"
    diff_threshold = 0.1  # Threshold for inconsistency detection

    # Evaluate inter-frame consistency
    results = evaluate_inter_frame_consistency(image_folder, gt_folder, diff_threshold=diff_threshold)

    # Print and save results
    print("Evaluation Results:")
    for result in results:
        status = "INCONSISTENT" if result['is_inconsistent'] else "CONSISTENT"
        print(f"Frame1: {result['frame1']} | Frame2: {result['frame2']} | Image Diff: {result['image_diff']:.4f} | GT Diff: {result['gt_diff']:.4f} | Status: {status}")

    # Optionally, save results to a CSV file
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv("inter_frame_consistency_results.csv", index=False)
    print("Results saved to inter_frame_consistency_results.csv")




