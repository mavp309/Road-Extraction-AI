import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('.', '..')))
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------------------------------------
# Streamlit Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Road Extraction Inference",
    layout="wide",
)

# Small CSS tweak for hackathon look
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1720; color: #e6eef8; }
    .block-container { padding: 1rem 2rem; }
    .stHeader { color: #ffffff; }
    h1 { color: #ffffff; font-family: 'Inter', sans-serif; }
    .stImage img { border-radius: 8px; }
    .stSidebar { background-color: #071029; }
    </style>
    """,
    unsafe_allow_html=True,
)

# session state for storing latest probability map and metadata
if "prob_map" not in st.session_state:
    st.session_state["prob_map"] = None
    st.session_state["model_name"] = None
    st.session_state["device_name"] = None
    st.session_state["last_upload_name"] = None
    st.session_state["saved_mask_path"] = None
    st.session_state["G_mask"] = None
    st.session_state["G_healed"] = None

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose page",
    ["Segmentation", "Graph Healing"],
    index=0,
)

if page == "Segmentation":
    st.sidebar.header("Segmentation Controls")
    show_heatmap = st.sidebar.checkbox("Show confidence heatmap", value=True)
    threshold = st.sidebar.slider("Binary threshold", 0.0, 1.0, 0.5, 0.01)
    line_thickness = st.sidebar.slider("Fallback line thickness", 2, 20, 6)
else:
    show_heatmap = True
    threshold = 0.5
    line_thickness = 6

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops.model_inference import RoadSegmenter

# ----------------------------------------------------------
# Model and file paths
# ----------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "mlops" / "models" / "best_model.pth"
MASK_SAVE_PATH = PROJECT_ROOT / "frontend" / "saved_mask.png"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    return RoadSegmenter(checkpoint_path=str(MODEL_PATH))


try:
    segmenter = load_model()
except FileNotFoundError as exc:
    segmenter = None
    st.warning(str(exc))
    st.warning("Using fallback synthetic road network generator")


# ----------------------------------------------------------
# Road mask helpers
# ----------------------------------------------------------
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


def blend_overlay_from_prob(image, prob_map, tint=(0, 255, 0), tint_strength=0.6):
    img_f = image.astype(np.float32)
    alpha = prob_map.astype(np.float32)
    if alpha.ndim == 2:
        alpha = np.expand_dims(alpha, axis=2)
    color = np.array(tint, dtype=np.float32)
    blended = img_f * (1.0 - tint_strength * alpha) + color * (tint_strength * alpha)
    return blended.astype(np.uint8)


def plot_graph_preview(G, title="Graph"):
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = {n: (data["x"], data["y"]) for n, data in G.nodes(data=True)}
    nx.draw(
        G,
        pos=pos,
        node_size=2,
        edge_color="#66ff66",
        width=0.5,
        alpha=0.6,
        ax=ax,
        linewidths=0,
    )
    ax.set_facecolor("#111111")
    ax.set_title(title, color="white")
    ax.axis("off")
    fig.tight_layout()
    return fig


def load_graph_helpers():
    try:
        from graph import build_graph_from_saved_mask
        from heal import heal_topological_gaps
        return build_graph_from_saved_mask, heal_topological_gaps
    except Exception as exc:
        st.error(f"Graph helper import failed: {exc}")
        return None, None


def render_segmentation_page():
    st.title("Occlusion-Robust Road Extraction")
    st.markdown(
        """
Upload a satellite image.

The uploaded image is passed directly to the trained DeepLabV3 model,
which predicts a binary road segmentation mask.
"""
    )

    uploaded_file = st.file_uploader(
        "Upload Satellite Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    if uploaded_file is None:
        st.info("Upload an image to start segmentation.")
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    uploaded_name = getattr(uploaded_file, "name", None)
    if st.session_state.get("last_upload_name") != uploaded_name:
        st.session_state["prob_map"] = None
        st.session_state["model_name"] = None
        st.session_state["device_name"] = None
        st.session_state["last_upload_name"] = uploaded_name

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_rgb, use_container_width=True)

    if st.button("Run Extraction Model", type="primary"):
        with st.spinner("Running DeepLabV3 Inference..."):
            if segmenter is None:
                st.info("Model is not available. Using a synthetic fallback road network.")
                prob_map = create_fallback_road_mask(original_image, thickness=line_thickness).astype(np.float32) / 255.0
                model_name = "Fallback (synthetic road network)"
                device_name = "cpu"
            else:
                prob_map = segmenter.predict(original_image, as_probability=True)
                model_name = "DeepLabV3-ResNet50"
                device_name = str(segmenter.device)

        st.session_state["prob_map"] = prob_map
        st.session_state["model_name"] = model_name
        st.session_state["device_name"] = device_name

    if st.session_state.get("prob_map") is None:
        with col2:
            st.info("Click 'Run Extraction Model' to compute a prediction.")
        return

    prob_map = st.session_state["prob_map"]
    if prob_map.shape[0:2] != original_image.shape[0:2]:
        prob_map = cv2.resize(prob_map, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        st.session_state["prob_map"] = prob_map

    binary_mask = (prob_map > threshold).astype(np.uint8)
    display_mask = (binary_mask * 255).astype(np.uint8)
    overlay = segmenter.overlay(original_image, prob_map) if segmenter is not None else blend_overlay_from_prob(original_image, prob_map, tint=(0, 255, 0), tint_strength=0.6)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    prob_uint8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    target_w, target_h = original_rgb.shape[1], original_rgb.shape[0]
    overlay_resized = cv2.resize(overlay_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = cv2.resize(heatmap_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    display_mask_resized = cv2.resize(display_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    with col2:
        st.subheader("Predicted Road Mask")
        st.image(display_mask_resized, use_container_width=True, clamp=True)

        overlay_col, heatmap_col = st.columns([1, 1])
        with overlay_col:
            st.subheader("Road Overlay")
            st.image(overlay_resized, use_container_width=True)
        with heatmap_col:
            if show_heatmap:
                st.subheader("Confidence Heatmap")
                st.image(heatmap_resized, use_container_width=True)

        if st.button("Save current mask for healing"):
            cv2.imwrite(str(MASK_SAVE_PATH), display_mask_resized)
            st.session_state["saved_mask_path"] = str(MASK_SAVE_PATH)
            st.success(f"Saved mask for graph healing at {MASK_SAVE_PATH}")

        st.success("Inference Complete!")

        with st.expander("Image Metadata"):
            st.write(f"Original Image Shape : {original_image.shape}")
            st.write(f"Predicted Mask Shape : {display_mask_resized.shape}")
            st.write(f"Model : {st.session_state.get('model_name')}")
            st.write(f"Device : {st.session_state.get('device_name')}")


def render_graph_healing_page():
    st.title("Graph Healing")
    st.markdown(
        """
Load the saved mask from the Segmentation page, convert it into a skeleton graph, and run topological healing to reconnect missing road segments.
"""
    )

    if not MASK_SAVE_PATH.exists():
        st.warning("No saved mask found. First save a mask on the Segmentation page.")
        return

    build_graph_from_saved_mask, heal_topological_gaps = load_graph_helpers()
    if build_graph_from_saved_mask is None:
        return

    if st.button("Load saved mask graph"):
        try:
            G_mask = build_graph_from_saved_mask(MASK_SAVE_PATH)
            st.session_state["G_mask"] = G_mask
            st.session_state["G_healed"] = None
            st.success("Loaded saved mask and built the skeleton graph.")
        except Exception as exc:
            st.error(f"Unable to load saved mask graph: {exc}")

    if st.session_state.get("G_mask") is None:
        st.info("Loaded graph will appear here after pressing 'Load saved mask graph'.")
        return

    G_mask = st.session_state["G_mask"]
    st.write("### Mask-derived graph")
    st.write(f"Nodes: {len(G_mask.nodes())}")
    st.write(f"Edges: {len(G_mask.edges())}")
    if len(G_mask.nodes()) <= 2000:
        fig = plot_graph_preview(G_mask, title="Mask Graph")
        st.pyplot(fig)
    else:
        st.warning("Graph is large; preview skipped to keep the UI responsive.")

    if st.button("Run graph healing"):
        with st.spinner("Running graph healing..."):
            G_healed = heal_topological_gaps(G_mask, max_dist_meters=25, min_alignment=0.30, max_passes=3)
            st.session_state["G_healed"] = G_healed
            st.success("Graph healing completed.")

    if st.session_state.get("G_healed") is not None:
        G_healed = st.session_state["G_healed"]
        st.write("### Healed graph summary")
        st.write(f"Nodes: {len(G_healed.nodes())}")
        st.write(f"Edges: {len(G_healed.edges())}")
        if len(G_healed.nodes()) <= 2000:
            fig = plot_graph_preview(G_healed, title="Healed Graph")
            st.pyplot(fig)
        else:
            st.warning("Healed graph is large; preview skipped.")


if page == "Segmentation":
    render_segmentation_page()
else:
    render_graph_healing_page()
