import cv2
import numpy as np

def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the pixel value at the clicked coordinates (x, y)
        pixel_bgr = frame[y, x]
        
        # Convert BGR to HSV
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        
        # Print the HSV values
        print(f"HSV value at ({x}, {y}): {pixel_hsv}")

# Start video capture from the camera
cap = cv2.VideoCapture(0)

# Create a window and set the mouse callback function
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', get_hsv_value)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break
    
    # Show the camera feed
    cv2.imshow('Camera', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
