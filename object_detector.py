import cv2 # For image processing and display
from ultralytics import YOLO # Import the YOLO model class

# --- Step 1: Load the YOLOv8 model ---
# Replace 'Object_detector.pt' with the path to your actual pre-trained Object model file.
# If you don't have a specific Object model, you can download a general YOLOv8 model:
# Example for general model: model = YOLO('yolov8n.pt') # 'n' stands for nano, a small, fast model
try:
    model = YOLO('Object_detector.pt') # Attempt to load a custom Object model
    print("Object detector model loaded successfully!")
except Exception as e:
    print(f"Error loading Object_detector.pt: {e}")
    print("Attempting to load a general YOLOv8n model.")
    model = YOLO('yolov8n.pt') # Fallback to a general model
    print("General YOLOv8n model loaded.")


# --- Step 2: Load the image for detection ---
image_path = 'Sample\image1.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}. Check the file path.")
else:
    print(f"Image '{image_path}' loaded for detection.")

    # --- Step 3: Perform object detection (inference) ---
    # model.predict() runs the detection on the image.
    # 'conf=0.5' sets the confidence threshold (only show detections with >50% confidence).
    # 'iou=0.7' sets the Intersection Over Union threshold for non-maximum suppression (removes duplicate boxes).
    # 'save=True' (optional) saves the image with detections to a 'runs/detect' folder.
    # 'show=False' (optional) prevents the model from automatically opening a display window.
    results = model.predict(source=img, conf=0.8, iou=0.7, save=False, show=False)

    # --- Step 4: Process and visualize the results ---
    # The 'results' object contains information about detected objects.
    # Each result object corresponds to an image (we have one here).
    for r in results:
        # r.boxes contains all detected bounding boxes
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get coordinates (x1,y1 top-left; x2,y2 bottom-right)
            confidence = round(box.conf[0].item(), 2) # Get confidence score
            cls = int(box.cls[0].item()) # Get class ID
            class_name = model.names[cls] # Get class name from model's class list

            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1) # Green rectangle

            # Put label (class name and confidence) above the box
            label = f"{class_name}: {confidence}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

            print(f"Detected: {class_name} at [{x1},{y1},{x2},{y2}] with confidence {confidence}")

    # --- Step 5: Display the image with detections ---
    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Detection script finished.")