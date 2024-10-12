from PIL import Image
import numpy as np
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

# Path to the input image
image_path = "wrapped_square_image2.png"  # Replace with your image path
color_matrix = divide_image_and_assign_colors(image_path)

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
