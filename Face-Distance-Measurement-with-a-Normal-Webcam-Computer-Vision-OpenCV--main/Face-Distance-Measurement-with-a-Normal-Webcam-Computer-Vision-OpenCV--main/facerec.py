import cv2
import numpy as np

# Load pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the video capture is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define variables to store previous frame's face area and a list for recent face areas
prev_face_area = None
face_area_history = []
moving_toward_camera = False

# Parameters for detecting moving toward camera
AREA_THRESHOLD = 100  # Minimum change in area to consider as moving
HISTORY_SIZE = 15  # Number of frames to track movement

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame with adjusted scaleFactor for distant faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    # Check if any face was detected
    if len(faces) > 0:
        # For simplicity, use the largest detected face (closest to the camera)
        (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])  # Choose the face with the largest area
        current_face_area = w * h  # Calculate face area

        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Add the current face area to the history
        face_area_history.append(current_face_area)
        if len(face_area_history) > HISTORY_SIZE:
            face_area_history.pop(0)  # Keep only the most recent frames

        # Calculate the average face area change over the recent history for smoother movement detection
        if len(face_area_history) == HISTORY_SIZE:
            avg_face_area = np.mean(face_area_history)

            # Check if we have a previous face area to compare with
            if prev_face_area is not None:
                # Compare the average face area with the previous face area for more accuracy
                if avg_face_area > prev_face_area + AREA_THRESHOLD:
                    moving_toward_camera = True
                    cv2.putText(img, "Moving Toward Camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    moving_toward_camera = False
                    cv2.putText(img, "Person detected ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update the previous face area with the current average area
            prev_face_area = avg_face_area

    # Display the resulting frame
    cv2.imshow("Face Detection", img)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
