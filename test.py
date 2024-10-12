import cv2
import numpy as np
import os
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# Map black to 1, white to 0, and undefined areas to -1
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
    return (value < 0.7) and (saturation < 0.799)  # Low value and low saturation for black

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
        if count / total_pixels > 0.43:  # Check for 50% dominance
            return color
    return '-1'  # Return '-1' if no dominant color

def divide_image_and_assign_colors(image_path, grid_size=25):
    # Open the image
    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure image is in RGB mode
    
    # Resize image to a grid
    img = img.resize((grid_size * 25, grid_size * 25))

    output_matrix = []
    for y in range(0, grid_size * 25, grid_size):
        row = []
        for x in range(0, grid_size * 25, grid_size):
            # Crop the grid cell
            cell = img.crop((x, y, x + grid_size, y + grid_size))
            # Get the dominant color for the cell
            dominant_color = get_dominant_color(np.array(cell))
            row.append(dominant_color)
        output_matrix.append(row)
    
    return np.array(output_matrix)

# Convert the color matrix to a numeric matrix using the mapping
def convert_to_numeric_matrix(color_matrix):
    numeric_matrix = np.vectorize(lambda color: color_to_number.get(color, -1))(color_matrix)
    return numeric_matrix

# Function to find objects with blue borders in the image
def find_blue_border_object(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_object = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and area > max_area:
            largest_object = approx
            max_area = area

    return largest_object

# Function to wrap the image from its corners into a square
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

# Step 1: Capture video, detect blue-border object, and save the cropped image
def detect_and_crop_blue_border_object():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.05
        blue_border_object = find_blue_border_object(frame, min_area)

        if blue_border_object is not None:
            x, y, w, h = cv2.boundingRect(blue_border_object)
            cropped_image = frame[y:y+h, x:x+w]
            output_image_path = os.path.join(os.getcwd(), 'cropped_blue_border_object.png')
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Cropped image saved at: {output_image_path}")
            cap.release()
            cv2.destroyAllWindows()
            return output_image_path  # Return the path to the cropped image
        else:
            print("No blue-border object detected in this frame.")
        
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

# Step 2: Use the cropped image as input and detect outer blue border, then wrap to square
def detect_outer_blue_border_and_wrap(input_image_path):
    frame = cv2.imread(input_image_path)
    if frame is None:
        print(f"Failed to load image: {input_image_path}")
        return

    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * 0.05
    wrapped_image = wrap_to_square(frame)

    # Save wrapped square image
    square_output_image_path = os.path.join(os.getcwd(), 'wrapped_square_image.png')
    cv2.imwrite(square_output_image_path, wrapped_image)
    print(f"Wrapped square image saved at: {square_output_image_path}")
    
    return square_output_image_path

def main():
    # Step 1: Detect and crop blue border object
    cropped_image_path = detect_and_crop_blue_border_object()

    if cropped_image_path:
        # Step 2: Detect outer blue border and wrap to square using the cropped image
        wrapped_image_path = detect_outer_blue_border_and_wrap(cropped_image_path)
        
        # Step 3: Process the wrapped image using the color detection method
        if wrapped_image_path:
            # Path to the input image
            color_matrix = divide_image_and_assign_colors(wrapped_image_path)

            # Display the output color matrix
            print("Color Matrix:")
            for row in color_matrix:
                print(row)

            # Convert color matrix to numeric matrix
            numeric_matrix = convert_to_numeric_matrix(color_matrix)

            # Display the numeric matrix
            print("Numeric Matrix:")
            print(numeric_matrix)

            # Optional: Visualize the result using a heatmap
            plt.imshow(numeric_matrix, cmap='gray')
            plt.show()

if __name__ == "__main__":
    main()
