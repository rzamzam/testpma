import streamlit as st
from keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from util import classify, set_background

# Set background
set_background('./BG/bg1.jpg')

# Set title
st.title('Chest X-Ray Classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model_path = './model/PMA.h5'

try:
    model = load_model(model_path)

    # Check if the model is compiled
    if not model._is_compiled:
        # Compile the model
        opt = Adam(learning_rate=0.000001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        st.info("Model compiled successfully.")

except OSError as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

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
