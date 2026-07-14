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
### Deployed on https://huggingface.co/spaces/mavp309/Road-Extraction-AI
## Running Locally

If you want to run this application on your local machine instead of the Hugging Face server, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Boot up the Streamlit interface:

   ```bash
   streamlit run app.py
   ```

## Repository Structure
```bash
.
├── app.py
├── frontend
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-314.pyc
│   │   ├── preprocess.cpython-314.pyc
│   │   ├── sidebar.cpython-314.pyc
│   │   └── state.cpython-314.pyc
│   ├── saved_mask.png
│   ├── sidebar.py
│   ├── state.py
│   └── test_ui.py
├── graphs
│   ├── dashboard.py
│   ├── graph.py
│   ├── heal.py
│   ├── __init__.py
│   ├── mask_processing.py
│   ├── simulation.py
│   ├── topology.py
│   └── visualise.py
├── mlops
│   ├── __init__.py
│   ├── model_inference.py
│   ├── models
│   │   └── best_model.pth
│   └── __pycache__
│       ├── __init__.cpython-314.pyc
│       └── model_inference.cpython-314.pyc
├── README.md
└── requirements.txt
```
