import numpy as np
import cv2

def generate_colored_image_from_matrix(matrix, colors, output_path= 'colored_grid.png'):
    # Create an empty image with the same size as the matrix
    colored_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
    
    # Map the matrix values to the color palette
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            colored_image[i, j] = colors[matrix[i, j]]
    
    # Save the colored image
    cv2.imwrite(output_path, colored_image)
    
    return colored_image

original_matrix = np.array([
    [0, 0, 0, 0, 4],
    [4, 0, 2, 0, 0],
    [0, 0, 0, 0, 0],
    [4, 0, 2, 1, 0],
    [0, 0, 0, 0, 0]
])

colored_image = generate_colored_image_from_matrix(original_matrix, 5)