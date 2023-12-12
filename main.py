import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

from util import classify, set_background

set_background('./BG/bg.jpg')

# set title
st.title('Pneumonia covid classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier without compiling
model = tf.keras.models.load_model('./model/PMAFP.h5', compile=False)

# Manually compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
