import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
# You can use other YOLOv8 models like 'yolov8s.pt', 'yolov8m.pt', etc. 
# for better accuracy at the cost of speed. 'yolov8n.pt' is the fastest.
model_path = "yolov8n.pt" 
# The number '0' tells OpenCV to use the default webcam. 
# If you have multiple webcams, you can try '1', '2', etc.
webcam_id = 0

# --- INITIALIZATION ---

# Initialize the YOLOv8 model
print(f"Loading YOLOv8 model: {model_path}...")
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Open the webcam feed
print(f"Accessing webcam (ID: {webcam_id})...")
cap = cv2.VideoCapture(webcam_id)
if not cap.isOpened():
    print(f"Error: Could not open webcam with ID {webcam_id}.")
    exit()

print("Starting webcam feed. Press 'q' to exit.")
frame_count = 0

# --- VIDEO PROCESSING LOOP ---
while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        frame_count += 1
        
        # 1. DETECT: Run YOLOv8 inference on the frame
        # The `model()` call returns a list of results objects.
        results = model(frame)

        # 2. VISUALIZE: Get the annotated frame with detections drawn on it.
        # The results object has a `plot()` method that handles all the drawing
        # of bounding boxes and labels for you.
        annotated_frame = results[0].plot()

        # 3. DISPLAY: Show the annotated frame in a window
        cv2.imshow("YOLOv8 Webcam Object Recognition", annotated_frame)

        # Allow the user to quit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("User pressed 'q'. Exiting...")
            break
    else:
        # This might happen if the webcam is disconnected.
        print("Failed to capture frame from webcam. Exiting.")
        break

# --- CLEANUP ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("Program finished.")
