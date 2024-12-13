import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import requests # or gdown
from io import BytesIO # added to handle file content
import gdown

# ... (other imports) ...

# Function to Load the Model from Cloud Storage
def load_model_from_cloud(url):
    """Loads the pre-trained model from cloud storage."""
    # Replace the shareable Google Drive link with a direct download link
    # This can be achieved using the 'id' from your shareable link.
    file_id = url.split('/')[-2] # Extract file id from the link
    
    # Download the file using gdown
    gdown.download(id=file_id, output="plant_disease_prediction_model.h5", quiet=False)

    model = tf.keras.models.load_model("plant_disease_prediction_model.h5")
    return model

# Replace with your shareable link
model_url = "https://drive.google.com/file/d/1-6JXzYS7O4F_rvFHd91ABQ52YK4SIrlX/view?usp=sharing" # example shareable link
model = load_model_from_cloud(model_url) 

with open('class_indices.json') as f:
    class_indices = json.load(f)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.set_page_config(page_title="Plant Disease Predictor", page_icon="🌿")
st.title('Plant Disease Predictor')

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Upload an image of a plant leaf.
    2. Click the 'Predict' button to predict the disease.
    """
)
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Predict'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
