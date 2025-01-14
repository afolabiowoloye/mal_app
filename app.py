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
from streamlit_option_menu import option_menu # for setting up menu bar
import cv2
import zipfile


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
#st.title("Plasmo3Net: Malaria Detection using CNN")

#-----------Web page setting-------------------#
page_title = "💊Plasmo3Net: Malaria Detection using CNN"
page_icon = "🎗🧬⌬"
viz_icon = "📊"
stock_icon = "📋"
picker_icon = "👇"
layout = "centered"
#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)

# Title of the app
#st.title("pIC50 Prediction App")
# Logo image
#image = 'logo/logo.jpg'
#st.image(image, use_container_width=True)


def process_image(uploaded_file):
    # Load the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the red blood cells
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours of the segmented red blood cells
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a directory to save the individual red blood cell images
    if not os.path.exists('cells'):
        os.makedirs('cells')

    # Save each red blood cell as a separate image file
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cell_image = image[y:y+h, x:x+w]
        cv2.imwrite(f'cells/cell_{i+1}.png', cell_image)

    # Delete images less than 1KB (noise)
    for filename in os.listdir('cells'):
        file_path = os.path.join('cells', filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) < 10024:
            os.remove(file_path)

    return contours

# Function to remove background and save images
def remove_background_and_save():
    input_dir = './cells'
    output_dir = './RBC'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = os.listdir(input_dir)
    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            cell_image = image[y:y+h, x:x+w]
            cell_mask = np.zeros_like(image)
            cv2.drawContours(cell_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            cell_mask = cv2.resize(cell_mask, (cell_image.shape[1], cell_image.shape[0]))
            cell_image = cv2.bitwise_and(cell_image, cell_mask)
            cell_filename = f'imagemdx_{idx}_{i:02d}.png'
            cell_path = os.path.join(output_dir, cell_filename)
            cv2.imwrite(cell_path, cell_image)

    # Delete images less than 1KB (noise)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) < 10024:
            os.remove(file_path)

# Function to create a ZIP file of the segmented images
def create_zip_of_images(output_dir):
    zip_filename = "segmented_cells.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    return zip_filename




selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'Cell segmentation', 'Prediction', 'About'],
    icons = ["house-fill", "capsule", "heart-fill", "heart"],
    default_index = 0,
    orientation = "horizontal"
)

if selected == "Home":
    st.subheader("Welcome to Malaria...")
    st.write("This application is designed to assist researchers and healthcare professionals in predicting...")
   

if selected == "Prediction":
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


if selected == "Cell segmentation":	
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption='Uploaded Image', use_container_width=True)

        # Reset the uploaded file pointer for processing
        uploaded_file.seek(0)

        # Process the image and count contours
        contours = process_image(uploaded_file)
        st.write(f"Found {len(contours)} red blood cell(s).")
    
        # Remove background and save images
        remove_background_and_save()
    
        # Create ZIP file of the segmented images
        if os.path.exists('./RBC'):
            zip_file_path = create_zip_of_images('./RBC')
        
            # Provide a download button for the ZIP file
            with open(zip_file_path, "rb") as f:
                st.download_button(
                    label="Download All Segmented Images",
                    data=f,
                    file_name=zip_file_path,
                    mime="application/zip")

# %%
