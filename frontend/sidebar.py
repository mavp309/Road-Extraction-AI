import streamlit as st

def render_sidebar():
    """Renders the sidebar and returns configuration parameters."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Example controls based on typical road extraction workflows
        model_selection = st.selectbox(
            "Select Model", 
            ["DeepLabV3+", "UNet"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
        
        st.markdown("---")
        st.info("Upload an image in the main panel to begin.")
        
        return {
            "model": model_selection,
            "threshold": confidence_threshold
        }