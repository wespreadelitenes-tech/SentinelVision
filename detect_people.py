import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

# YOLOv8n pre-trained model (COCO dataset - 80 classes including 'person')
model = YOLO('yolov8n.pt')

# Trajectory points for each tracked person
tracks = {}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera nahi khul rahi! Check karo camera connected hai ya nahi.")
    exit()

print("✅ Camera khul gayi! People detection chal rahi hai...")
print("🔴 Q dabao band karne ke liye")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame nahi mil raha!")
        break

    # YOLO detection — class 0 = person
    results = model.predict(frame, conf=0.5, classes=[0], verbose=False)

    # Annotated frame with boxes
    annotated = results[0].plot()

    # Count people
    num_people = len(results[0].boxes)

    # Info overlay
    cv2.putText(annotated, f"People Detected: {num_people}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, "Press Q to Quit", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("People Detection - YOLOv8", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
