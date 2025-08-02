import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort # Make sure sort.py is in the same directory

# --- CONFIGURATION ---
# To use a video file, put the path here. To use your webcam, set it to 0.
video_path = "Sample\sample_video_highway.mp4" # 0 for webcam
model_path = "yolov8n.pt" # Using the nano model for speed

# These are the COCO classes for vehicles. We will only track these.
VEHICLE_CLASSES = [2, 3, 5, 7] # 2: car, 3: motorcycle, 5: bus, 7: truck

# --- INITIALIZATION ---

# Initialize YOLO model
print("Loading YOLO model...")
model = YOLO(model_path)
print("Model loaded.")

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Open video file or webcam
print(f"Opening video source: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video source {video_path}")
    exit()

# Create a dictionary to store colors for each track ID
track_colors = {}

# --- PROCESSING LOOP ---
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or webcam disconnected.")
        break

    frame_num += 1
    
    # 1. DETECT: Run YOLOv8 inference on the frame
    detections = model(frame, verbose=False)[0]

    # 2. PREPARE FOR TRACKER: Filter and format detections
    detections_for_sort = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in VEHICLE_CLASSES and score > 0.4:
            detections_for_sort.append([x1, y1, x2, y2, score])
    
    detections_for_sort = np.array(detections_for_sort)

    # 3. TRACK: Update the SORT tracker with the new detections
    tracked_objects = tracker.update(detections_for_sort)

    # 4. VISUALIZE: Draw bounding boxes and IDs on the frame
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        # Assign a unique color to each track ID
        if track_id not in track_colors:
            track_colors[track_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        color = track_colors[track_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('Phase 1: Vehicle Tracking', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# --- CLEANUP ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()