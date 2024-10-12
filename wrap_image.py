import cv2
import numpy as np
import os

# Function to find objects with outer blue borders in the image
def find_outer_blue_border_object(frame, min_area):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for detecting blue color (outer blue border)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Optional: Denoise the mask to remove small noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_object = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Approximate the contour to get the shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the current contour is the largest one
        if area > max_area:
            largest_object = approx
            max_area = area

    return largest_object

def main():
    # Path to the input image
    input_image_path = 'cropped_blue_border_object.png'  # Replace with your input image path
    frame = cv2.imread(input_image_path)
    
    if frame is None:
        print(f"Failed to load image: {input_image_path}")
        return

    # Define the minimum area for detecting the object with an outer blue border
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * 0.05  # Adjust this threshold based on the image size

    # Find the object with an outer blue border
    blue_border_object = find_outer_blue_border_object(frame, min_area)

    if blue_border_object is not None:
        # Check if the contour has more or fewer than 4 points, approximate it to 4
        if len(blue_border_object) != 4:
            epsilon = 0.05 * cv2.arcLength(blue_border_object, True)
            blue_border_object = cv2.approxPolyDP(blue_border_object, epsilon, True)

        # Ensure we have exactly 4 points for the perspective transform
        if len(blue_border_object) == 4:
            # Get the bounding rectangle around the detected object
            x, y, w, h = cv2.boundingRect(blue_border_object)

            # Define points for the perspective transformation (source points are outer blue border)
            src_points = np.float32(blue_border_object.reshape(-1, 2))  # Points from the detected object
            dst_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])  # Corners of the new square

            # Calculate the perspective transform matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Perform the perspective warp
            warped_image = cv2.warpPerspective(frame, matrix, (w, h))

            # Save the new image
            output_image_path = os.path.join(os.getcwd(), 'wrapped_outer_blue_border.png')
            cv2.imwrite(output_image_path, warped_image)
            print(f"Wrapped outer blue border image saved at: {output_image_path}")
        else:
            print("Could not approximate to exactly 4 points.")
    else:
        print("No outer blue-border object detected in the image.")

if __name__ == "__main__":
    main()
