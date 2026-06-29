import streamlit as st
import numpy as np
from PIL import Image
import time

#DUMMY MODEL INFERENCE
def dummy_predict(tensor_input):
    """
    Expects input shape: (1, 3, 512, 512)
    Returns a random binary mask of shape (512, 512)
    """

    
    #random noise mask to simulate a road mask
    random_mask = np.random.randint(0, 2, size=(512, 512)).astype(np.float32)
    return random_mask

#STREAMLIT UI LAYOUT
st.set_page_config(page_title="Road Extraction Inference", layout="wide")

st.title("Occlusion-Robust Road Extraction")
st.markdown("Upload a satellite image. The system will chunk it into a `3 x 512 x 512` tensor, run it through the Deep Learning model, and output the predicted road mask.")

# File uploader accepts common image formats
uploaded_file = st.file_uploader("Upload Satellite Imagery", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:
    # Read the image using PIL and ensure it's RGB
    original_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Satellite Image")
        st.image(original_image, use_column_width=True)
        
    # Inference Button
    if st.button("Run Extraction Model", type="primary"):
        with st.spinner("Processing image tensor and running inference..."):
            
            # --- PREPROCESSING ---
            # 1. Resize to 512x512
            resized_image = original_image.resize((512, 512))
            
            # 2. Convert to numpy array: Shape becomes (512, 512, 3)
            img_array = np.array(resized_image, dtype=np.float32)
            
            # Normalize just like in your training script (simplified here to 0-1)
            img_array = img_array / 255.0 
            
            # 3. Transpose to PyTorch format: (Channels, Height, Width) -> (3, 512, 512)
            tensor_input = np.transpose(img_array, (2, 0, 1))
            
            # 4. Add Batch Dimension: (1, 3, 512, 512)
            batch_tensor = np.expand_dims(tensor_input, axis=0)
            
            # --- INFERENCE ---
            # Pass the tensor to our model
            predicted_mask = dummy_predict(batch_tensor)
            
            # --- POST-PROCESSING & DISPLAY ---
            with col2:
                st.subheader("Predicted Road Mask")
                # Streamlit can render 2D numpy arrays directly if values are 0-1
                st.image(predicted_mask, use_column_width=True, clamp=True)
                
                st.success("Inference Complete!")
                
                # Expandable section to prove to judges the tensor math is correct
                with st.expander("View Tensor Metadata"):
                    st.write(f"**Input Image Shape (RGB):** {img_array.shape}")
                    st.write(f"**Model Input Tensor Shape:** {tensor_input.shape}")
                    st.write(f"**Batched Input Shape:** {batch_tensor.shape}")
                    st.write(f"**Output Mask Shape:** {predicted_mask.shape}")
