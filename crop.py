import cv2
import numpy as np

def find_blue_border(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for the blue color
    lower_blue = np.array([90, 50, 50])  # Adjust if necessary
    upper_blue = np.array([140, 255, 255])
    
    # Create a mask for blue
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Denoise and improve the mask
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.erode(mask, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0

    # Find the largest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    return largest_contour

def crop_image_from_blue_border(frame, contour):
    if contour is not None:
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the image using the bounding box
        cropped_image = frame[y:y+h, x:x+w]

        return cropped_image
    return None

def main():
    # Load the image file instead of capturing video
    image_path = 'check2.jpeg'  # Change this to the path of your input image
    frame = cv2.imread(image_path)

    if frame is None:
        print("Failed to load image.")
        return
    
    # Find the blue border
    blue_border_contour = find_blue_border(frame)

    if blue_border_contour is not None:
        # Crop the image from the blue border
        cropped_image = crop_image_from_blue_border(frame, blue_border_contour)
        
        if cropped_image is not None:
            # Save the cropped image
            cv2.imwrite('cropped_image.png', cropped_image)
            print("Cropped image saved!")

    # Display the original frame without the contour
    cv2.imshow("Original Frame", frame)

    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
