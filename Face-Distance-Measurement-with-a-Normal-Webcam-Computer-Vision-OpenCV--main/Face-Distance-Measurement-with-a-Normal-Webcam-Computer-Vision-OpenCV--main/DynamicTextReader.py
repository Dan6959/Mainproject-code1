import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import requests
import time

# Discord Webhook URL (replace this with your own webhook URL)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1293256325215813752/0WGWQOET7bc6KhqZhxe52ttCgeIDvxLimWEZzDjFKrvdqOs81NbEXtlE_i8jSY7wppe_"

# Change camera index if 0 doesn't work; 0 is the default
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be initialized")
    exit()

detector = FaceMeshDetector(maxFaces=1)

threshold_distance = 200  # Distance threshold in cm
door_open = False
door_locked = False
unlock_time = 0
alert_cooldown = 10  # Cooldown time in seconds for sending alerts

def send_discord_alert(image, message):
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {
        'file': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')
    }
    data = {
        "content": message
    }
    response = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
    if response.status_code == 204:
        print("Alert sent to Discord successfully.")
    else:
        print("Failed to send alert to Discord.")

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to grab frame")
        break

    imgText = np.zeros_like(img)  # Blank image for the text
    img, faces = detector.findFaceMesh(img, draw=False)  # Face mesh detection

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)  # Distance between two face points
        W = 6.3  # Approximate width of face in cm

        # Calculate the distance (depth) of the face from the camera
        f = 1000  # Focal length (adjust as necessary for calibration)
        d = (W * f) / w  # Depth equation
        print(f"Distance: {d} cm")

        # Display the depth information on the image
        cvzone.putTextRect(img, f'Distance: {int(d)} cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

        current_time = time.time()
        
        # Check if person is within threshold distance and door is currently locked
        if d < threshold_distance and not door_open and not door_locked:
            # Unlock the door and start countdown
            door_open = True
            unlock_time = current_time  # Start the countdown
            cvzone.putTextRect(img, "Door unlocked", (50, 50), scale=2, colorR=(0, 255, 0))
            send_discord_alert(img, "⚠️ Alert: Door unlocked as person crossed the threshold distance!")
        
        # Lock the door if the person remains within threshold after countdown
        elif door_open and current_time - unlock_time >= 10:
            if d < threshold_distance:
                door_open = False
                door_locked = True
                cvzone.putTextRect(img, "Door locked", (50, 50), scale=2, colorR=(0, 0, 255))
                send_discord_alert(img, "🔒 Door locked as person remained in range!")
            else:
                # Person moved back out of range, reset to locked state
                door_open = False
                door_locked = False

        # Condition to re-unlock only after moving out of range and coming back within threshold
        elif d >= threshold_distance:
            # Person moved out of range, allow re-unlocking if they come back in
            door_locked = False

        # Display door status on the frame
        if door_open:
            cvzone.putTextRect(img, "Door unlocked", (50, 50), scale=2, colorR=(0, 255, 0))
        elif door_locked:
            cvzone.putTextRect(img, "Door locked", (50, 50), scale=2, colorR=(0, 0, 255))

    # Show the final image
    cv2.imshow("Image", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
