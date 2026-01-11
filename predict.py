# Import necessary libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("./models/mask_detector_model.h5")
print("Model loaded successfully.")

# Open webcam for real-time prediction
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_resized = np.expand_dims(face_resized, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(face_resized)
        label = "Mask" if prediction[0][0] > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display warning message if no mask
        if label == "No Mask":
            cv2.putText(frame, "WARNING: Please wear a mask!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("Face Mask Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
