import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('.')))
import streamlit as st
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops.model_inference import RoadSegmenter


def create_fallback_road_mask(image, thickness=6):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    edge_mask = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=2)

    bright = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)

    mask = cv2.bitwise_or(edge_mask, bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
    mask = cv2.medianBlur(mask, 7)

    if np.count_nonzero(mask) < 0.02 * mask.size:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        max_radius = min(width, height) // 2 - 10

        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x1 = int(center[0] + np.cos(angle) * max_radius)
            y1 = int(center[1] + np.sin(angle) * max_radius)
            cv2.line(mask, center, (x1, y1), 255, thickness=thickness)

        branch_points = [
            (width // 4, height // 4),
            (3 * width // 4, height // 4),
            (width // 4, 3 * height // 4),
            (3 * width // 4, 3 * height // 4),
        ]

        for point in branch_points:
            cv2.line(mask, center, point, 255, thickness=max(2, thickness // 2))

        cv2.circle(mask, center, max(10, max_radius // 5), 255, thickness=-1)
        mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)

    return mask


def create_fallback_overlay(image, mask):
    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)


# ----------------------------------------------------------
# Load Model (Only Once)
# ----------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "mlops" / "models" / "best_model.pth"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}"
        )

    return RoadSegmenter(checkpoint_path=str(MODEL_PATH))


try:
    segmenter = load_model()
except FileNotFoundError as exc:
    segmenter = None
    st.warning(str(exc))
    st.warning("Using fallback synthetic road network generator")


# ----------------------------------------------------------
# Streamlit Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Road Extraction Inference",
    layout="wide"
)

st.title("Occlusion-Robust Road Extraction")

st.markdown(
    """
Upload a satellite image.

The uploaded image is passed directly to the trained DeepLabV3 model,
which predicts a binary road segmentation mask.
"""
)


# ----------------------------------------------------------
# Upload Image
# ----------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)


# ----------------------------------------------------------
# Inference
# ----------------------------------------------------------
if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    original_image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    original_rgb = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            original_rgb,
            use_container_width=True
        )

    if st.button("Run Extraction Model", type="primary"):

        with st.spinner("Running DeepLabV3 Inference..."):
            if segmenter is None:
                st.info("Model is not available. Using a synthetic fallback road network.")
                predicted_mask = create_fallback_road_mask(original_image)
                overlay = create_fallback_overlay(original_image, predicted_mask)
                model_name = "Fallback (synthetic road network)"
                device_name = "cpu"
            else:
                predicted_mask = segmenter.predict(original_image)
                overlay = segmenter.overlay(
                    original_image,
                    predicted_mask
                )
                model_name = "DeepLabV3-ResNet50"
                device_name = segmenter.device

            overlay = cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            )

            display_mask = predicted_mask
            if display_mask.dtype == np.uint8 and display_mask.max() <= 1:
                display_mask = (display_mask * 255).astype(np.uint8)

        with col2:

            st.subheader("Predicted Road Mask")

            st.image(
                display_mask,
                use_container_width=True,
                clamp=True
            )

            st.subheader("Road Overlay")

            st.image(
                overlay,
                use_container_width=True
            )

            st.success("Inference Complete!")

            with st.expander("Image Metadata"):

                st.write(f"Original Image Shape : {original_image.shape}")

                st.write(f"Predicted Mask Shape : {predicted_mask.shape}")

                st.write(f"Model : {model_name}")

                st.write(f"Device : {device_name}")