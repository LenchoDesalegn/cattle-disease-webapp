import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('cattle_disease_webapp/cattle_disease_model.h5')
custom_objects={
        'Functional': tf.keras.Model,       # required for functional API models
        'MobileNetV2': MobileNetV2          # pretrained layer used
    }

class_names = ['Normal Skin', 'Lumpy Skin']
first_aid = {
    "Normal Skin": "No action needed. Continue regular checkups and vaccination schedule.",
    "Lumpy Skin": "Isolate affected cattle, clean lesions, apply antiseptic, and consult a veterinarian."
}

st.title("🐄 Cattle Disease Diagnosis Web App")

# Upload image
uploaded_file = st.file_uploader("Upload a cattle skin image", type=["jpg","jpeg","png"])

# Capture from webcam
capture_file = st.camera_input("Or capture image from webcam")

img = None
if uploaded_file:
    img = Image.open(uploaded_file)
elif capture_file:
    img = Image.open(capture_file)

if img:
    st.image(img, caption="Selected Image", width=300)
    
    # Preprocess image
    img_array = np.array(img.resize((192,192)))/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    prediction = class_names[idx]
    suggestion = first_aid[prediction]
    
    # Show results
    st.success(f"Prediction: {prediction}")
    st.info(f"First Aid Suggestion: {suggestion}")


