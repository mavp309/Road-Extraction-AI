import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from frontend.state import init_session_state
from frontend.sidebar import render_sidebar
from mlops.model_inference import RoadSegmenter
from graphs.graph import build_graph_from_saved_mask
from graphs.heal import heal_topological_gaps

# ----------------------------------------------------------
# Project paths
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "mlops" / "models" / "best_model.pth"
MASK_SAVE_PATH = PROJECT_ROOT / "frontend" / "saved_mask.png"

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Road Extraction & Healing",
    layout="wide",
)

# Custom CSS for the dark hackathon look
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1720; color: #e6eef8; }
    .block-container { padding: 1rem 2rem; }
    .stHeader { color: #ffffff; }
    h1, h2, h3, h4 { color: #ffffff; font-family: 'Inter', sans-serif; }
    .stImage img { border-radius: 8px; }
    .stSidebar { background-color: #071029; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Road Extraction & Graph Healing")
st.write(
    "Upload an aerial image, run road segmentation, save the predicted mask, "
    "and build/heal the derived road graph using the tabs below."
)

# Initialize state and sidebar
init_session_state()
config = render_sidebar()

# ----------------------------------------------------------
# Model utilities
# ----------------------------------------------------------
@st.cache_resource
def load_segmenter():
    if not MODEL_PATH.exists():
        return None
    return RoadSegmenter(checkpoint_path=str(MODEL_PATH))

segmenter = load_segmenter()


def prob_map_to_mask_image(prob_map, threshold):
    mask = (prob_map > threshold).astype(np.uint8) * 255
    return mask


def save_mask_image(mask_image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask_image)
    return str(path)


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


def render_inference_tab():
    st.header("Image Upload & Road Masking")
    uploaded_file = st.file_uploader("Upload Satellite/Aerial Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["image"] = image_rgb

    if st.session_state.get("image") is None:
        st.info("Upload an image to start.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state["image"], use_column_width=True)

    with col2:
        st.subheader("Inference")
        if st.button("Run Inference", type="primary"):
            with st.spinner("Processing inference..."):
                image_bgr = cv2.cvtColor(st.session_state["image"], cv2.COLOR_RGB2BGR)
                if segmenter is None:
                    st.warning("Model checkpoint not found. Using fallback synthetic mask.")
                    prob_map = np.zeros((st.session_state["image"].shape[0], st.session_state["image"].shape[1]), dtype=np.float32)
                else:
                    prob_map = segmenter.predict(image_bgr, as_probability=True)
                st.session_state["prob_map"] = prob_map
                st.success("Inference complete.")

        if st.session_state.get("prob_map") is not None:
            prob_map = st.session_state["prob_map"]
            mask_image = prob_map_to_mask_image(prob_map, config["threshold"])
            st.session_state["mask_image"] = mask_image

            st.write("### Mask Preview")
            st.image(mask_image, clamp=True, use_column_width=True)

            if st.button("Save mask for graph construction"):
                save_mask_image(mask_image, MASK_SAVE_PATH)
                st.session_state["saved_mask_path"] = str(MASK_SAVE_PATH)
                st.success(f"Saved mask to {MASK_SAVE_PATH}")

            if segmenter is not None:
                overlay = segmenter.overlay(image_bgr, prob_map)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                st.write("### Confidence Overlay")
                st.image(overlay, use_column_width=True)

                heatmap = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                st.write("### Probability Heatmap")
                st.image(heatmap, use_column_width=True)


def render_graph_construction_tab():
    st.header("Mask-Derived Graph")

    if st.session_state.get("prob_map") is None and st.session_state.get("saved_mask_path") is None:
        st.info("Please run inference and save the mask in Step 1 first.")
        return

    if st.button("Generate Skeleton Graph"):
        with st.spinner("Building graph from mask..."):
            if st.session_state.get("mask_image") is None:
                prob_map = st.session_state["prob_map"]
                mask_image = prob_map_to_mask_image(prob_map, config["threshold"])
                st.session_state["mask_image"] = mask_image
                save_mask_image(mask_image, MASK_SAVE_PATH)
                st.session_state["saved_mask_path"] = str(MASK_SAVE_PATH)

            G_mask = build_graph_from_saved_mask(MASK_SAVE_PATH)
            st.session_state["G_mask"] = G_mask
            st.success("Skeleton graph generated from saved mask.")

    if st.session_state.get("G_mask") is None:
        st.info("Generate the skeleton graph to continue.")
        return

    G_mask = st.session_state["G_mask"]
    col_stats, col_viz = st.columns([1, 2])
    with col_stats:
        st.write("### Graph Statistics")
        st.metric("Nodes", len(G_mask.nodes()))
        st.metric("Edges", len(G_mask.edges()))

    with col_viz:
        fig = plot_graph_preview(G_mask, title="Mask Graph")
        st.pyplot(fig)
        st.write(
            "Large graphs may take a moment to render. If the preview is slow, "
            "try zooming or use the node/edge summary values above."
        )


def render_healing_tab():
    st.header("Heal Topological Gaps")

    if st.session_state.get("G_mask") is None:
        st.info("Please generate a graph in Step 2 first.")
        return

    if st.button("Run Graph Healing", type="primary"):
        with st.spinner("Running topology healing..."):
            G_healed = heal_topological_gaps(
                st.session_state["G_mask"],
                max_dist_meters=25,
                min_alignment=0.30,
                max_passes=3,
            )
            st.session_state["G_healed"] = G_healed
            st.success("Graph healing completed.")

    if st.session_state.get("G_healed") is None:
        st.info("Run healing to view the reconstructed network.")
        return

    G_healed = st.session_state["G_healed"]
    col_stats, col_viz = st.columns([1, 2])
    with col_stats:
        st.write("### Healed Summary")
        st.metric("Nodes", len(G_healed.nodes()))
        st.metric("Edges", len(G_healed.edges()))

    with col_viz:
        fig = plot_graph_preview(G_healed, title="Healed Graph")
        st.pyplot(fig)
        st.write(
            "Large healed graphs may take a moment to render. "
            "Use the summary metrics above to verify topology improvements."
        )


tab1, tab2, tab3 = st.tabs(["1. Inference", "2. Graph Construction", "3. Topology Healing"])

with tab1:
    render_inference_tab()

with tab2:
    render_graph_construction_tab()

with tab3:
    render_healing_tab()
