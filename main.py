import streamlit as st
from keras.optimizers import Adam
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


from util import classify, set_background

# Set background
set_background('./BG/bg.jpg')

# Set title
st.title('Pneumonia covid classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./model/PMA.h5')

# Compile the model
optimizer = Adam(lr=0.01)  # Adjust the learning rate as needed
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
