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

        if segmenter is None:
            st.error(
                "Model is not available. Please place the checkpoint at "
                f"{MODEL_PATH}"
            )
        else:
            with st.spinner("Running DeepLabV3 Inference..."):

                predicted_mask = segmenter.predict(original_image)

                overlay = segmenter.overlay(
                    original_image,
                    predicted_mask
                )

                overlay = cv2.cvtColor(
                    overlay,
                    cv2.COLOR_BGR2RGB
                )

            with col2:

                st.subheader("Predicted Road Mask")

                st.image(
                    predicted_mask,
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

                    st.write("Model : DeepLabV3-ResNet50")

                    st.write(f"Device : {segmenter.device}")