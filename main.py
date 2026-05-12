import os # file handling
import json # json processing
import numpy as np # numerical python
from PIL import Image # image access
import tensorflow as tf # Loads the pre-trained model and performs predictions.
import streamlit as st  # Builds the user interface for the web app.(library)
from prometheus_client import start_http_server, Counter # type: ignore
from prometheus_client import generate_latest # type: ignore

# Start the Prometheus metrics server on port 8000
start_http_server(8000)

# Load the pre-trained model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Initialize the prediction counter (ensure it's created only once)
if 'prediction_counter' not in st.session_state:
    st.session_state.prediction_counter = Counter('prediction_count', 'Total number of predictions made')

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    
    # Increment the prediction counter
    st.session_state.prediction_counter.inc()
    return predicted_class_name

# Streamlit app
st.title('Plant Disease Prediciton')
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Predict'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

# Display metrics at the end of the Streamlit app
st.sidebar.header("Metrics")
st.sidebar.write(f"Total Predictions Made: {st.session_state.prediction_counter._value.get()}")

# Display Prometheus metrics
if st.sidebar.button("Show Metrics"):
    metrics = generate_latest()
    st.sidebar.text(metrics.decode('utf-8'))
