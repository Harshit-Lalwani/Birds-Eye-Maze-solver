import cv2
import numpy as np
import os
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
    
    hue, saturation, value = pixel_hsv
    return (value < 0.69) and (saturation < 0.8)

def is_white(pixel):
    return pixel[0] > 200 and pixel[1] > 200 and pixel[2] > 200

def rgb_to_hsv(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    if delta == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / delta) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / delta) + 120) % 360
    elif max_c == b:
        h = (60 * ((r - g) / delta) + 240) % 360

    s = 0 if max_c == 0 else (delta / max_c)
    v = max_c

    return h, s, v

def get_dominant_color(cell_pixels):
    pixel_array = np.array(cell_pixels).reshape(-1, 3)

    color_counts = Counter()
    for pixel in pixel_array:
        if is_black(pixel):
            color_counts['black'] += 1
        elif is_white(pixel):
            color_counts['white'] += 1

    total_pixels = len(pixel_array)
    for color, count in color_counts.items():
        if count / total_pixels > 0.50:
            return color
    return '-1'

def divide_image_and_assign_colors(img, grid_size=25):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (grid_size * 25, grid_size * 25))

    output_matrix = []
    for y in range(0, grid_size * 25, grid_size):
        row = []
        for x in range(0, grid_size * 25, grid_size):
            cell = img[y:y + grid_size, x:x + grid_size]
            dominant_color = get_dominant_color(cell)
            row.append(dominant_color)
        output_matrix.append(row)
    
    return np.array(output_matrix)

def convert_to_numeric_matrix(color_matrix):
    numeric_matrix = np.vectorize(lambda color: color_to_number.get(color, -1))(color_matrix)
    return numeric_matrix

def find_blue_border(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.erode(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    return largest_contour

def crop_image_from_blue_border(frame, contour):
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = frame[y:y+h, x:x+w]
        return cropped_image
    return None

def find_outer_blue_border_object(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_object = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if area > max_area:
            largest_object = approx
            max_area = area

    return largest_object

def process_image_for_outer_blue_border(frame):
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * 0.05

    blue_border_object = find_outer_blue_border_object(frame, min_area)

    if blue_border_object is not None:
        if len(blue_border_object) != 4:
            epsilon = 0.05 * cv2.arcLength(blue_border_object, True)
            blue_border_object = cv2.approxPolyDP(blue_border_object, epsilon, True)

        if len(blue_border_object) == 4:
            x, y, w, h = cv2.boundingRect(blue_border_object)

            src_points = np.float32(blue_border_object.reshape(-1, 2))
            dst_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            warped_image = cv2.warpPerspective(frame, matrix, (w, h))
            return warped_image
    return None

def wrap_to_square(image, multiple_of=25):
    height, width, _ = image.shape

    src_points = np.float32([
        [0, 0],               
        [width - 1, 0],        
        [width - 1, height - 1],  
        [0, height - 1]        
    ])

    side_length = max(height, width)
    dst_points = np.float32([
        [0, 0],                     
        [side_length - 1, 0],        
        [side_length - 1, side_length - 1],  
        [0, side_length - 1]         
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    wrapped_image = cv2.warpPerspective(image, matrix, (side_length, side_length))

    if side_length % multiple_of != 0:
        final_size = side_length + (multiple_of - side_length % multiple_of)
        wrapped_image = cv2.resize(wrapped_image, (final_size, final_size))

    return wrapped_image

def main():
    image_path = 'check3.jpeg'
    frame = cv2.imread(image_path)

    if frame is None:
        return
    
    blue_border_contour = find_blue_border(frame)

    if blue_border_contour is not None:
        cropped_image = crop_image_from_blue_border(frame, blue_border_contour)
        if cropped_image is not None:
            wrapped_image = process_image_for_outer_blue_border(cropped_image)
            square_image = wrap_to_square(wrapped_image, multiple_of=25)
            color_matrix = divide_image_and_assign_colors(square_image)

            numeric_matrix = convert_to_numeric_matrix(color_matrix)
            plt.imshow(numeric_matrix, cmap='gray', interpolation='nearest')
            plt.axis('off')
            plt.show()
            # Output the numeric matrix
            for row in numeric_matrix.tolist():
                print(row)
                print()

if __name__ == "__main__":
    main()
