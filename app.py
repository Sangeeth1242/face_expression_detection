import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time

# Load the trained model
model = load_model('emotion_model.h5')

# Define the emotions corresponding to model classes
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the input image for prediction
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))  # Resize to match input size of model
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit App Layout
st.title("Real-Time Emotion Recognition from Webcam Feed")

st.write("This app uses your webcam to detect facial emotions in real-time.")

# Create a placeholder for the video stream
frame_placeholder = st.empty()

# Start video capture using OpenCV
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    st.error("Error: Could not access the webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break
        
        # Preprocess the frame for prediction
        preprocessed_frame = preprocess_image(frame)

        # Make prediction
        prediction = model.predict(preprocessed_frame)
        predicted_class = np.argmax(prediction[0])

        # Draw the predicted emotion on the frame
        emotion = EMOTIONS[predicted_class]
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Add a small delay to make the frame rate reasonable
        time.sleep(0.1)

# Release the webcam when the stream ends
cap.release()
