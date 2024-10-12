import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageColor

# Function to convert RGB to HSV
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

# Function to get pixel color at clicked location
def get_pixel_color(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        pixel_color = img[y, x]  # Get the RGB color from the image array
        hsv_color = rgb_to_hsv(pixel_color)  # Convert RGB to HSV
        print(f"Clicked at ({x}, {y}) - RGB: {pixel_color} - HSV: {hsv_color}")

# Load the image
image_path = "wrapped_square_image.png"  # Replace with your image path
img = Image.open(image_path).convert("RGB")
img = np.array(img)

# Display the image
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.connect('button_press_event', get_pixel_color)  # Connect click event
plt.title("Click on the image to get HSV values")
plt.show()
