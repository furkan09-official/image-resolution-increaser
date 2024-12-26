import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from keras.models import load_model


model = load_model('notebooks/super_resolution_model.h5')


print(os.path.exists('notebooks/super_resolution_model.h5'))


# Load the trained model
@st.cache_resource  # Cache the model to improve app performance
def load_super_resolution_model():
    return load_model('super_resolution_model.h5')
model = load_super_resolution_model()

# App title
st.title("Super-Resolution Image Generator")


# File upload
uploaded_file = st.file_uploader("Upload a low-resolution image (32x32)", type=["png", "jpg", "jpeg"])

# Process uploaded file
if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image_array = np.array(image)

    # Display the low-resolution image
    st.subheader("Low-Resolution Image")
    st.image(image, caption="Uploaded Low-Resolution Image", use_column_width=True)

    # Check if the input image is of size 32x32
    if image_array.shape != (32, 32):
        # Resize the image to 32x32 if it is not already
        image_array = cv2.resize(image_array, (32, 32), interpolation=cv2.INTER_AREA)
        st.warning("The image has been resized to 32x32 for processing.")

    # Preprocess the image for the model
    image_array = image_array.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Add batch and channel dimensions

    # Generate the super-resolution image
    sr_image = model.predict(image_array)[0, :, :, 0]  # Remove batch and channel dimensions
    sr_image = (sr_image * 255).clip(0, 255).astype(np.uint8)  # Convert back to [0, 255]

    # Display the super-resolution image
    st.subheader("Super-Resolution Image")
    st.image(sr_image, caption="Generated Super-Resolution Image (64x64)", use_column_width=True)

# Footer
st.write("---")
st.write("Built with ❤️ using Streamlit and TensorFlow")