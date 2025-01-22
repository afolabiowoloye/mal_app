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
#page_title1 = st.markdown("""
#    <h3 style='color: blue;'>💊Malaria Detection Web App using CNN</h3>
#    """, unsafe_allow_html=True)

page_title = "💊Malaria Detection Web App"
page_icon = "🎗🧬⌬"
viz_icon = "📊"
stock_icon = "📋"
picker_icon = "👇"
layout = "centered"
upload_icon ="📤"
segmentation_icon = "🔍"
classification_icon ="📊"
result_icon ="📈"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)


head_image = 'logo/header.png'
st.image(head_image, use_container_width=True)




selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'Cell segmentation', 'Run Diagnosis', 'About'],
    icons = ["house-fill", "capsule", "heart-fill", "heart"],
    default_index = 0,
    orientation = "horizontal"
)

if selected == "Home":
#    st.subheader("Welcome to Malaria Detection Web Application")
#    st.write("This application is designed to assist researchers and healthcare professionals in predicting...")
    st.markdown("""
    <h3 style='color: red;'>Welcome to Malaria Detection Web Application</h3>
    Welcome to our Malaria Detection Application! This innovative tool leverages advanced Convolutional Neural Network (CNN) architecture to assist in the early detection of malaria through the analysis of microscope slide images. Our application is designed to assist in the early detection of malaria through advanced image processing techniques and computer vision.
    
    <h5 style='color: blue;'>Here’s a step-by-step overview of how the application operates:</h5>  
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .streamlit-expanderHeader {background-color: #FAEBD7; color: black;}
    .streamlit-expanderContent {background-color: white;color: blue;}
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("**1. Upload Image**", icon=upload_icon):
        st.write("Users can upload microscope slide images containing red blood cells (RBCs). The application supports various image formats for ease of use.")

    with st.expander("**2. Segmentation**", icon=segmentation_icon):
        st.write("Once the image is uploaded, the application employs sophisticated image processing techniques to segment the RBCs from the background. This step is crucial for isolating the cells for further analysis.")
    
    with st.expander("**3. Classification**", icon=classification_icon):
        st.write("After segmentation, the application analyzes the isolated RBC images. Using a trained CNN model, it classifies each cell as either infected or uninfected. This classification is based on features learned during training, ensuring high accuracy.")
       
    with st.expander("**4. Results Display**", icon=result_icon):
        st.write("The results are displayed to the user, indicating the number of infected and uninfected RBCs in the uploaded image. This information is vital for healthcare professionals to diagnose malaria quickly and efficiently.")

    st.markdown("""
    <h6 style='color: blue;'>User-Friendly Interface</h6>
    This application is designed with a user-friendly interface, making it accessible to medical professionals and researchers. Users can easily navigate through the process and obtain results in real-time.
    """, unsafe_allow_html=True)
    
    with st.sidebar.header(<h3 style='color: red;'>How to Use the Web App</h3>):
          st.sidebar.markdown("""
          <h3 style='color: red;'>How to Use the Web App</h3> 
          **Step 1: Image Segmentation (For Microscope Slide Images)**
          
          i. If you are working with **microscope slide images**, click the **"Segmentation"** button.
          ii. The algorithm will automatically segment your images into individual **Red Blood Cell (RBC) images**
          
          **Step 2: Run Diagnosis (For Segmented RBC Images)**
          i. After segmentation (or if you already have **segmented RBC images**), click the **"Run Diagnosis"** button.
          ii. Upload your segmented RBC images for classification.
          iii. The classification results will be displayed in **real-time**.
          
          **Note**:\n
          If you are working with **pre-segmented RBC images**, skip **Step 1** and go directly to **Step 2** for classification.)
          """, unsafe_allow_html=True)
    


   

if selected == "Run Diagnosis":
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
        
###*** For single images***###
# File uploader
    #uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    #if uploaded_file is not None:
    # Load the test image
        #test_image = Image.open(uploaded_file)
    # Preprocess the test image
        #test_image = test_image.resize((img_width, img_height))
        #test_image = np.array(test_image) / 255.0  # Normalize the pixel values
        #test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension    
    # Make the prediction
        #prediction = model.predict(test_image)    
    # Interpret the prediction
        #label = "Uninfected" if prediction[0][0] >= 0.5 else "Parasitized"    
    # Display the image with the prediction
        #st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        #st.markdown(f"<h4 style='color: red;'>Predicted label: <strong>{label}</strong></h4>", unsafe_allow_html=True)
        ##st.write(f"Predicted Label: **{label}**")    
    # Optionally, save the predicted image
        #plt.imshow(test_image[0])
        #plt.title("Predicted Label: " + label)
        #plt.axis("off")
        #plt.savefig("predicted_image.png")
        #plt.close()  # Close the plot to avoid display issues
###*** End_For single images***###

# File uploader
    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
        # Load the test image
            test_image = Image.open(uploaded_file)        
        # Preprocess the test image
            test_image_resized = test_image.resize((img_width, img_height))
            test_image_array = np.array(test_image_resized) / 255.0  # Normalize the pixel values
            test_image_array = np.expand_dims(test_image_array, axis=0)  # Add batch dimension        
        # Make the prediction
            prediction = model.predict(test_image_array)        
        # Interpret the prediction
            label = "Uninfected" if prediction[0][0] >= 0.5 else "Parasitized"        
        # Display the image with the prediction
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            st.markdown(f"<h4 style='color: red;'>Predicted label: <strong>{label}</strong></h4>", unsafe_allow_html=True)        
        # Optionally, save the predicted image
            plt.imshow(test_image_array[0])
            plt.title("Predicted Label: " + label)
            plt.axis("off")
            plt.savefig(f"predicted_image_{uploaded_file.name}.png")
            plt.close()  # Close the plot to avoid display issues



### Beginning of functions
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
### End of functions



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
