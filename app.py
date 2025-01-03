#!/usr/bin/env python
# coding: utf-8
# %%
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

# Load the trained model from Google Drive
model_link_id = "1Fwhl5iEqgsoblKymdUFbp10eesb69O7T"
model_link = f'https://drive.google.com/uc?id={model_link_id}'

# Download the model file
model_filename = 'model.h5'
if not os.path.exists(model_filename):
    response = requests.get(model_link)
    with open(model_filename, 'wb') as f:
        f.write(response.content)

model = load_model(model_filename)

# Image dimensions
img_width = 64
img_height = 64

# Streamlit app title
st.title("Plasmo3Net: Malaria Detection using CNN")


image = 'logo/CNN_confusion_matrix_test.png'
st.image(image, use_container_width=True)
st.markdown("""
#### `` Figure 1: Testset Confusion Matrix``
""")

st.markdown("""
<h3 style='color: red;'>Model Evaluation</h3>
<h5 style='color: blue;'>Accuracy = 99.3%</h5>
<h5 style='color: blue;'>Precision = 99.1</h5>
<h5 style='color: blue;'>F1 score = 99.3</h5>
<h5 style='color: blue;'>Recall = 99.6</h5>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar.header('Instruction on how to use this app'):
      st.sidebar.markdown("""
This section will guide you on how to use this app..... some text here)
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the test image
    test_image = Image.open(uploaded_file)
    
    # Preprocess the test image
    test_image = test_image.resize((img_width, img_height))
    test_image = np.array(test_image) / 255.0  # Normalize the pixel values
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    
    # Make the prediction
    prediction = model.predict(test_image)
    
    # Interpret the prediction
    label = "Uninfected" if prediction[0][0] >= 0.5 else "Parasitized"
    
    # Display the image with the prediction
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    st.markdown(f"<h4 style='color: red;'>Predicted label: <strong>{label}</strong></h4>", unsafe_allow_html=True)
    #st.write(f"Predicted Label: **{label}**")
    
    # Optionally, save the predicted image
    plt.imshow(test_image[0])
    plt.title("Predicted Label: " + label)
    plt.axis("off")
    plt.savefig("predicted_image.png")
    plt.close()  # Close the plot to avoid display issues



# %%
