import cv2
import time
from ultralytics import YOLO
import threading
import requests

# --- Configuration ---
ip_stream_url = "http://192.168.137.211:81/stream"  # Change to your IP stream URL
model = YOLO("yolov8l-oiv7.pt")  # Load your YOLO model
announcement_interval = 5  # seconds to wait before re-announcing the same object

def send_to_server(announcement):
    """Send the detection result to the local Flask server."""
    url = "http://127.0.0.1:5000/update"  # Update if needed
    data = {"text": announcement}
    try:
        response = requests.post(url, json=data)
        print("Sent to server:", response.json())
    except Exception as e:
        print("Failed to send to server:", e)

def send_to_server_async(announcement):
    """Send detection data asynchronously so that network delays don't block the main loop."""
    threading.Thread(target=send_to_server, args=(announcement,), daemon=True).start()

# --- Open Video Stream ---
cap = cv2.VideoCapture(ip_stream_url)
#cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
wait = 0.2
# Tracking announcements
last_announced = {}      # Dictionary to store last announcement time for each object
prev_announcements = set()  # Set of announcements from the previous scene
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection on the current frame
    # results = model(frame,conf = 0.4)
    results = model(frame,conf = 0.3)
    annotated_frame = results[0].plot()
    frame_width = frame.shape[1]
    current_announcements = set()

    # Gather announcements from detections
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Format: [x1, y1, x2, y2]
        classes = results[0].boxes.cls.cpu().numpy()   # Class indices

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0

            # Determine horizontal position (left, center, right)
            if center_x < frame_width / 3:
                position = "left"
            elif center_x < 2 * frame_width / 3:
                position = "center"
            else:
                position = "right"

            class_id = int(cls)
            label = results[0].names.get(class_id, f"class_{class_id}")
            announcement = f"{label} on the {position}"
            if(label!="Clothing"):
                current_announcements.add(announcement)
            

    current_time = time.time()

    # If the scene has changed, announce the new scene immediately.
    if current_announcements != prev_announcements:
        if current_announcements:
            combined_announcement = "Detected: " + ", ".join(sorted(current_announcements))
            print("New scene:", combined_announcement)
            send_to_server_async(combined_announcement)
            for ann in current_announcements:
                last_announced[ann] = current_time
        else:
            send_to_server_async("No objects detected")
        prev_announcements = current_announcements.copy()
    else:
        # Scene remains the same: only re-announce objects if they haven't been announced in the past interval.
        new_announcements = []
        for announcement in current_announcements:
            last_time = last_announced.get(announcement, 0)
            if current_time - last_time > announcement_interval:
                new_announcements.append(announcement)
                last_announced[announcement] = current_time
        if new_announcements:
            combined_announcement = "Detected: " + ", ".join(sorted(new_announcements))
            print("Re-announcing:", combined_announcement)
            if(count % 10 == 0):
                send_to_server_async(combined_announcement)
                count = count + 1
            else:
                continue

    # Optionally, display the annotated frame locally (for debugging)
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
