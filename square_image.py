import cv2
import numpy as np
import os

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

def main():
    # Path to the input image (replace with your image path)
    input_image_path = 'wrapped_outer_blue_border.png'  # Replace with your image path
    image = cv2.imread(input_image_path)
    
    if image is None:
        print(f"Failed to load image: {input_image_path}")
        return
    
    # Perform the perspective warp to make the image square
    square_image = wrap_to_square(image, multiple_of=4)
    
    # Save the wrapped square image
    output_image_path = os.path.join(os.getcwd(), 'wrapped_square_image.png')
    cv2.imwrite(output_image_path, square_image)
    
    print(f"Wrapped square image saved at: {output_image_path}")

if __name__ == "__main__":
    main()
