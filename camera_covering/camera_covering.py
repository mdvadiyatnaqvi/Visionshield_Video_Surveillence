# 2926
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = keras.models.load_model(
    "./camera_covering_model.h5"
)  # Provide the path to your trained model


# Define a function to preprocess and predict on camera frames
def predict_camera_covering(frame):
    # Preprocess the frame
    frame = cv2.resize(frame, (128, 128))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0  # Rescale pixel values to [0, 1]

    # Make a prediction/mean value
    value = cv2.mean(frame)
    return value


# Open the camera
# You can change the index if you have multiple cameras
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mean value of real time camera covering
    prediction = predict_camera_covering(frame)
    # print(prediction)
    # Set a threshold for camera covering detection (adjust as needed)
    # threshold = 1.9824834e-35 #original
    threshold = 0.45
    # threshold = 15.9824834e-9999

    # Display the prediction result on the frame
    # if prediction[0][0] > threshold:
    if prediction[0] < threshold:
        text = "covered"
        color = (0, 0, 255)  # Red
        # cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # print('camera is convered')
    else:
        text = "Uncovered"
        color = (0, 255, 0)  # Green
        # cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # print('camera is uncovered')
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame
    cv2.imshow("Camera Covering Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
