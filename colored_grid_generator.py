import numpy as np
import cv2

def generate_colored_image_from_matrix(matrix, colors, output_path='colored_grid.png'):
    # Create an empty image with the same size as the matrix
    colored_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
    
    # Map the matrix values to the color palette
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            colored_image[i, j] = colors[matrix[i, j]]
    
    # Save the colored image
    cv2.imwrite(output_path, colored_image)
    
    return colored_image

def add_border_to_matrix(matrix, border_value, border_thickness=1):
    # Get the size of the original matrix
    rows, cols = matrix.shape
    
    # Create a new matrix with the border
    bordered_matrix = np.full((rows + 2 * border_thickness, cols + 2 * border_thickness), border_value, dtype=matrix.dtype)
    
    # Copy the original matrix into the center of the new matrix
    bordered_matrix[border_thickness:border_thickness + rows, border_thickness:border_thickness + cols] = matrix
    
    return bordered_matrix

# Define the color palette (black, white, blue, red, green)
colors = [
    [0, 0, 0],       # Black
    [255, 255, 255], # White
    [0, 0, 255],     # Blue
    [255, 0, 0],     # Red
    [0, 255, 0]      # Green
]

original_matrix = np.array([
    [0, 0, 0, 0, 4],
    [4, 0, 2, 0, 0],
    [0, 0, 0, 0, 0],
    [4, 0, 2, 1, 0],
    [0, 0, 0, 0, 0]
])

# Add a blue border (represented by 2) to the matrix
bordered_matrix = add_border_to_matrix(original_matrix, border_value=2)

# Generate the colored image from the bordered matrix
colored_image = generate_colored_image_from_matrix(bordered_matrix, colors)

# Save the generated image and exit
output_path = 'colored_grid_with_border.png'
cv2.imwrite(output_path, colored_image)
print(f"Image saved as {output_path}")