import streamlit as st
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

# Load TFLite model
@st.cache_resource  # caches the model
def load_model():
    interpreter = tflite.Interpreter(model_path="cattle_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names and first aid
class_names = ['Normal Skin', 'Lumpy Skin']
first_aid = {
    "Normal Skin": "No action needed. Continue regular checkups and vaccination schedule.",
    "Lumpy Skin": "Isolate affected cattle, clean lesions, apply antiseptic, and consult a veterinarian."
}

st.title("🐄 Cattle Disease Diagnosis Web App (TFLite)")

# Upload image
uploaded_file = st.file_uploader("Upload a cattle skin image", type=["jpg","jpeg","png"])

# Capture from webcam
capture_file = st.camera_input("Or capture image from webcam")

# Process selected image
img = None
