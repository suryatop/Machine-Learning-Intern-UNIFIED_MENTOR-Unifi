import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#  Load the trained model
model_path = "/Users/suryatopsasmal/Downloads/Projects/Animal Classification/animal_classifier.h5"
model = tf.keras.models.load_model(model_path)

#  Define class labels
class_labels = ["Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant", "Giraffe", 
                "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"]

#  Streamlit UI
st.title("🐾 Animal Image Classifier")
st.write("Upload an image to classify the animal.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    #  Preprocess the image
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    #  Predict
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    st.write(f"###  Prediction: {predicted_class}")
