import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from lime import lime_image
from tensorflow.keras.models import load_model
from explainability import explain_with_lime, explain_with_gradcam

# Load the CNN model
@st.cache(allow_output_mutation=True)
def load_cnn_model():
    return tf.keras.models.load_model('models/cnn_model.h5')

# Function to preprocess uploaded image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    return image

# Main function to run the app
def main():
    st.title('Food Allergen Detection')

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image of food...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Load the model
        cnn_model = load_cnn_model()

        # Button to classify the image
        if st.button('Classify'):
            # Predict using the model
            st.write("Classifying...")
            prediction = cnn_model.predict(np.expand_dims(processed_image, axis=0))
            classes = ['No Allergens', 'Contains Allergens']  # Replace with your actual classes
            predicted_class = classes[np.argmax(prediction)]

            # Display prediction
            st.success(f'Prediction: {predicted_class}')

            # Explainability: LIME
            st.subheader('LIME Explanation')
            explain_with_lime(cnn_model, processed_image)

            # Explainability: Grad-CAM
            st.subheader('Grad-CAM Visualization')
            explain_with_gradcam(cnn_model, processed_image)

# Run the app
if __name__ == '__main__':
    main()

