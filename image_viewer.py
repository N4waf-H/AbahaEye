import cv2 # Import the OpenCV library

# --- Step 1: Load the image ---
# cv2.imread() reads an image from the specified file path.
# Make sure 'street.jpg' is in the same directory as your script,
# or provide the full path to the image.
image = cv2.imread('Sample\image1.jpg')

# --- Step 2: Check if the image was loaded successfully ---
if image is None:
    print("Error: Could not load image. Check the file path.")
else:
    print("Image loaded successfully!")

    # --- Step 3: Display the image ---
    # cv2.imshow() displays an image in a window.
    # The first argument is the window name (you can choose any string).
    # The second argument is the image matrix itself.
    cv2.imshow('Image Viewr', image)

    # --- Step 4: Wait for a key press and close the window ---
    # cv2.waitKey(0) waits indefinitely until a key is pressed.
    # If you put a number (e.g., 1000), it waits for that many milliseconds.
    cv2.waitKey(0)

    # cv2.destroyAllWindows() closes all OpenCV windows.
    cv2.destroyAllWindows()

print("Script finished.")