import streamlit as st

def init_session_state():
    """Initialize all required session state variables."""
    default_states = {
        "prob_map": None,
        "model_name": None,
        "device_name": None,
        "last_upload_name": None,
        "saved_mask_path": None,
        "G_mask": None,
        "G_healed": None,
        "image": None, # Store the uploaded image
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value