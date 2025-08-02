import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
# IMPORTANT: Change this to the path of your video file
video_path = "Sample\sample_video_highway.mp4" 
# You can use other YOLOv8 models like 'yolov8s.pt', 'yolov8m.pt', etc. for better accuracy at the cost of speed.
model_path = "yolov8n.pt" 
# The name of the output video file that will be created
output_video_path = "recognition_output.mp4"

# --- INITIALIZATION ---

# Initialize the YOLOv8 model
print(f"Loading YOLOv8 model: {model_path}...")
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Open the video file
print(f"Opening video file: {video_path}...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at '{video_path}'")
    exit()

# Get video properties (width, height, frames per second) for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
# 'mp4v' is a good, common codec for .mp4 files.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("Starting video processing...")
frame_count = 0

# --- VIDEO PROCESSING LOOP ---
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1
        print(f"Processing frame {frame_count}...")
        
        # 1. DETECT: Run YOLOv8 inference on the frame
        # The `model()` call returns a list of results objects.
        results = model(frame)

        # 2. VISUALIZE: Get the annotated frame with detections drawn on it.
        # The results object has a `plot()` method that handles all the drawing
        # of bounding boxes and labels for you. It's very convenient.
        annotated_frame = results[0].plot()

        # 3. DISPLAY: Show the annotated frame in a window
        # This is optional and can be commented out if you only want to save the file.
        cv2.imshow("YOLOv8 Object Recognition", annotated_frame)

        # 4. SAVE: Write the annotated frame to the output video file
        out.write(annotated_frame)

        # Allow the user to quit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("User interrupted the process.")
            break
    else:
        # We've reached the end of the video
        print("End of video file.")
        break

# --- CLEANUP ---
print("Releasing resources...")
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete.")
print(f"Annotated video has been saved to: {output_video_path}")
