import cv2
import time

# Use camera index 0 (default webcam) and the DirectShow backend for Windows
CAP_INDEX = 0
BACKEND = cv2.CAP_DSHOW

print(f"Attempting to open camera index {CAP_INDEX} with backend {BACKEND}...")
cap = cv2.VideoCapture(CAP_INDEX, BACKEND)

# Check if the camera initialized correctly
if not cap.isOpened():
    print("FATAL ERROR: Could not open camera. Check permissions and drivers.")
else:
    print("Camera object successfully created. Flushing buffer...")

    # --- Buffer Flushing Section ---
    # Wait 2 seconds for the camera to warm up and clear initial black/junk frames
    time.sleep(2)
    for _ in range(5):
        cap.read()
    print("Buffer cleared. Starting video stream.")

    # --- Main Display Loop ---
    while cap.isOpened():
        # Read a new frame from the camera
        success, frame = cap.read()
        
        if not success:
            print("Failed to capture image frame. Breaking loop.")
            break
        
        # Display the frame in a window named 'Camera Test Feed'
        cv2.imshow('Camera Test Feed', frame)
        
        # Check for user input every 1 millisecond
        # If the 'q' key is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Cleanup ---
# Release the camera resource
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
print("Program finished and windows closed.")
