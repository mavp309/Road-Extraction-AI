---
title: Road Extraction AI
emoji: 🛰️
colorFrom: indigo
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Occlusion-Robust Road Extraction

This is a deep learning application deployed on Hugging Face Spaces that performs semantic segmentation on satellite imagery to extract road networks. It is specifically trained to identify roads even under severe occlusions, such as cloud cover or dense tree canopies.

## Model Architecture

- **Architecture:** U-Net
- **Encoder:** ResNet-34 (Pre-trained on ImageNet)
- **Framework:** PyTorch & `segmentation_models_pytorch`
- **Augmentations:** Features custom occlusion simulation using `Albumentations` (CoarseDropout) to improve robustness against real-world interference.

## Running Locally

If you want to run this application on your local machine instead of the Hugging Face server, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have your model weights (`road_segmentation_unet.pth`) placed in the root directory.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Boot up the Streamlit interface:

   ```bash
   streamlit run app.py
   ```

## Repository Structure

- `app.py`: The main Streamlit web application.
- `train_segmentation.py`: The PyTorch training pipeline and dataset definitions.
- `requirements.txt`: Python dependencies (configured for CPU deployment).
- `README.md`: Hugging Face Space configuration and documentation.
