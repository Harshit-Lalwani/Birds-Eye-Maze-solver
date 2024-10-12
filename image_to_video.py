import cv2
import numpy as np
import os

# Function to find objects with blue borders in the image
def find_blue_border_object(frame, min_area):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for detecting blue color in the borders (adjusted for more precision)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Optional: Denoise the mask to remove small noise
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_object = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Approximate the contour to get the shape
        epsilon = 0.01 * cv2.arcLength(contour, True)  # More precise contour approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Assuming the object has 4 sides (quadrilateral shape)
        if len(approx) == 4 and area > max_area:
            largest_object = approx
            max_area = area

    return largest_object

def main():
    cap = cv2.VideoCapture(0)

    # Reduce the resolution of the frames for faster processing (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Define the minimum area for detecting the object with a blue border
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.05  # Adjust this threshold based on the image size

        # Find the object with a blue border
        blue_border_object = find_blue_border_object(frame, min_area)

        if blue_border_object is not None:
            # Get the bounding rectangle around the detected object
            x, y, w, h = cv2.boundingRect(blue_border_object)

            # Crop the image to the bounding rectangle
            cropped_image = frame[y:y+h, x:x+w]

            # Save the cropped image with the detected blue-border object
            output_image_path = os.path.join(os.getcwd(), 'cropped_blue_border_object.png')
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Cropped image saved at: {output_image_path}")

            # Stop the video feed after finding the object
            cap.release()
            cv2.destroyAllWindows()
            return
        else:
            print("No blue-border object detected in this frame.")

        # Show the video feed
        cv2.imshow("Video Feed", frame)

        # Reduce delay to 1 millisecond for faster frame capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
