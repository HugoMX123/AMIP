# Image Segmentation for Autonomous Driving

## Overview
This project focuses on developing an image segmentation model for autonomous driving, specifically addressing the challenges posed by different weather conditions (sunny vs. rainy). The goal is to bridge the performance gap of models trained on sunny images when deployed in rainy scenarios, using the RaidaR dataset.

---

## Objectives

1. Develop and evaluate segmentation models.
2. Implement strategies to improve robustness.
3. Document experiments, findings, and methodologies.

---

## Dataset

### RaidaR Dataset
The RaidaR dataset contains annotated images designed to evaluate autonomous driving under rainy conditions. It includes diverse street scenes featuring challenges such as:
- Water droplets
- Reflections
- Fog

**Challenges:**
- The ground truth (GT) annotations for rainy images are often poor and inconsistent.
- GT masks were generated using state-of-the-art segmentation networks, but they contain inaccuracies, especially in rainy conditions.

---

## Methods

### Preprocessing
1. **Data Discarding:**
   - Used Laplacian variance to identify and remove noisy images.
   - Ensured only high-quality images were retained for training.

2. **Data Denoising:**
   - Applied Fourier Transform to filter out high-frequency noise.
   - Reconstructed denoised images while preserving essential details.

3. **Augmentation Techniques:**
   - Horizontal flipping.
   - Planned further exploration with rotation, scaling, random crops, and color/intensity changes.

4. **Dataset Splitting:**
   - Split into 70% training, 15% validation, and 15% test sets.
   - Ensured balanced class representation across splits.

5. **Resizing:**
   - Experimented with image sizes (256x256 and 128x128).
   - Primarily used 128x128 due to hardware constraints.

### Modeling
- Chose U-Net architecture with variations in depth:
  - 4-4 encoding-decoding blocks (best performance).
  - Compared with 3-3 and 5-5 block configurations.
- Loss functions:
  - Categorical Crossentropy.
  - Dice Loss (better performance in mean IoU).
- Optimizer: Adam with learning rates of $10^{-4}$, $10^{-3}$, and $10^{-2}$.
- Saved the best-performing models based on validation metrics.

### Evaluation Metrics
1. **Mean Intersection over Union (IoU):**
   \[ \text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c} \]
   \[ \text{Mean IoU} = \frac{1}{C} \sum_{c=1}^{C} \text{IoU}_c \]

2. **Dice Coefficient:**
   \[ \text{Dice}_c = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c} \]
   \[ \text{Mean Dice} = \frac{1}{C} \sum_{c=1}^{C} \text{Dice}_c \]

---

## Results

### Key Findings
- The 4-4 block U-Net outperformed other configurations.
- Dice Loss provided better mean IoU compared to Categorical Crossentropy.
- Downsizing images to 128x128 helped reduce noise and computational load.

### Denoising Observations
- Applying denoising to already downsized images limited its effectiveness.
- Poor quality ground truth data negatively impacted results.

### Example Visualizations
- Successful predictions.
- Poor predictions with noisy ground truth.

---

## Future Work
1. Improve pre-processing techniques, such as advanced augmentations and better denoising.
2. Explore transfer learning with pre-trained models on similar datasets.
3. Investigate the use of post-processing algorithms like Conditional Random Fields (CRFs).

---

## Reproducibility

- **Repository:** [GitHub Repository](https://github.com/HugoMX123/AMIP)
- **Models:** [Google Drive Folder](https://drive.google.com/drive/folders/1zOTCeVdRDuIpFEDm0iSGvtvQwGpGHhhV?usp=sharing)
- **Results Table:** [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1i4QMTx_8xaEJgEOI3L31WJO_pPJWOHjCio2ggKD8-y4/edit?usp=sharing)

---

## References
1. Jin, J. et al., RaidaR: A Rich Annotated Image Dataset of Rainy Street Scenes. [arXiv](https://arxiv.org/abs/2104.04606)
2. U-Net Architecture: [Original Paper](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
