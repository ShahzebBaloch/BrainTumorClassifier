import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image

model = load_model('tumor_classifier_model3.keras')
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Brain Tumor Classification")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((227, 227))
    st.image(img, caption="Uploaded MRI", use_container_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    st.success(f"Prediction: **{class_names[class_idx]}**")
