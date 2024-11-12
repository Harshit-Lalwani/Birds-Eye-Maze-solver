import cv2
import numpy as np
import os
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

color_to_number = {
    'black': 1,
    'white': 0,
    '-1': -1  # Undefined areas
}

def is_black(pixel):
    # Convert RGB to HSV
    pixel_hsv = rgb_to_hsv(pixel)
    
    # Define black thresholds based on your HSV observations
    hue, saturation, value = pixel_hsv
    # Conditions based on observed ranges, you can tweak these thresholds as necessary
    return (value < 0.69) and (saturation < 0.8)  # Low value and low saturation for black

def is_white(pixel):
    return pixel[0] > 200 and pixel[1] > 200 and pixel[2] > 200  # White threshold

def rgb_to_hsv(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    # Hue calculation
    if delta == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / delta) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / delta) + 120) % 360
    elif max_c == b:
        h = (60 * ((r - g) / delta) + 240) % 360

    # Saturation calculation
    s = 0 if max_c == 0 else (delta / max_c)

    # Value calculation
    v = max_c

    return h, s, v

def get_dominant_color(cell_pixels):
    # Convert pixels to an array and flatten
    pixel_array = np.array(cell_pixels).reshape(-1, 3)

    # Count occurrences of each color
    color_counts = Counter()
    for pixel in pixel_array:
        if is_black(pixel):
            color_counts['black'] += 1
        elif is_white(pixel):
            color_counts['white'] += 1

    total_pixels = len(pixel_array)
    for color, count in color_counts.items():
        if count / total_pixels > 0.50:  # Check for 50% dominance
            return color
    return '-1'  # Return '-1' if no dominant color

def divide_image_and_assign_colors(img, grid_size=25):
    # OpenCV loads the image in BGR format, convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to a grid
    img = cv2.resize(img, (grid_size * 25, grid_size * 25))

    output_matrix = []
    for y in range(0, grid_size * 25, grid_size):
        row = []
        for x in range(0, grid_size * 25, grid_size):
            # Crop the grid cell
            cell = img[y:y + grid_size, x:x + grid_size]
            # Get the dominant color for the cell
            dominant_color = get_dominant_color(cell)
            row.append(dominant_color)
        output_matrix.append(row)
    
    return np.array(output_matrix)

# Convert the color matrix to a numeric matrix using the mapping
def convert_to_numeric_matrix(color_matrix):
    numeric_matrix = np.vectorize(lambda color: color_to_number.get(color, -1))(color_matrix)
    return numeric_matrix

# First part: Cropping the image using the blue border
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

# Second part: Finding objects with outer blue borders
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

def process_image_for_outer_blue_border(frame):
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
            return warped_image;
            # Save the new image
            # output_image_path = os.path.join(os.getcwd(), 'wrapped_outer_blue_border.png')
            # cv2.imwrite(output_image_path, warped_image)
            # print(f"Wrapped outer blue border image saved at: {output_image_path}")
        else:
            print("Could not approximate to exactly 4 points.")
    else:
        print("No outer blue-border object detected in the image.")

def wrap_to_square(image, multiple_of=25):
    height, width, _ = image.shape

    # Define the original corner points (assuming the outer blue border)
    src_points = np.float32([
        [0, 0],               # Top-left corner
        [width - 1, 0],        # Top-right corner
        [width - 1, height - 1],  # Bottom-right corner
        [0, height - 1]        # Bottom-left corner
    ])

    # Define the destination points (for a perfect square)
    side_length = max(height, width)
    dst_points = np.float32([
        [0, 0],                     # Top-left corner
        [side_length - 1, 0],        # Top-right corner
        [side_length - 1, side_length - 1],  # Bottom-right corner
        [0, side_length - 1]         # Bottom-left corner
    ])

    # Perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the image to make it a perfect square
    wrapped_image = cv2.warpPerspective(image, matrix, (side_length, side_length))

    # Ensure the size is a multiple of 4
    if side_length % multiple_of != 0:
        final_size = side_length + (multiple_of - side_length % multiple_of)
        wrapped_image = cv2.resize(wrapped_image, (final_size, final_size))

    return wrapped_image

def main():
    # Load the image file instead of capturing video
    image_path = 'check.jpeg'  # Change this to the path of your input image
    frame = cv2.imread(image_path)

    if frame is None:
        print("Failed to load image.")
        return
    
    # Find the blue border and crop the image
    blue_border_contour = find_blue_border(frame)

    if blue_border_contour is not None:
        cropped_image = crop_image_from_blue_border(frame, blue_border_contour)
        if cropped_image is not None:
            # Save the cropped image
            # cropped_image_path = 'cropped_image.png'
            # cv2.imwrite(cropped_image_path, cropped_image)
            # print(f"Cropped image saved at: {cropped_image_path}")
            
            # Now process the cropped image for the outer blue border
            wrapped_image=process_image_for_outer_blue_border(cropped_image)
            square_image = wrap_to_square(wrapped_image, multiple_of=25)
            color_matrix = divide_image_and_assign_colors(square_image)

# Display the output color matrix
            print("Color Matrix:")
            for row in color_matrix:
                print(row)

            # Convert color matrix to numeric matrix
            numeric_matrix = convert_to_numeric_matrix(color_matrix)

            # Display the numeric matrix
            print("Numeric Matrix:")
            print(numeric_matrix)

            # Optional: Visualize the grid using the numeric matrix
            plt.imshow(numeric_matrix, cmap='gray', interpolation='nearest')
            plt.axis('off')
            plt.show()
        else:
            print("Failed to crop the image.")
    else:
        print("No blue border detected.")

if __name__ == "__main__":
    main()
