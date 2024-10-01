import unittest
import numpy as np
import cv2
from ESW.qr_to_matrix import qr_to_color_matrix

def generate_colored_image_from_matrix(matrix, colors, output_path):
    # Create an empty image with the same size as the matrix
    colored_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
    
    # Map the matrix values to the color palette
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            colored_image[i, j] = colors[matrix[i, j]]
    
    # Save the colored image
    cv2.imwrite(output_path, colored_image)
    
    return colored_image

class TestQRToColorMatrix(unittest.TestCase):

    def test_colored_qr_code_image(self):
        # Generate a matrix with values 0, 1, 2, 3, and 4
        original_matrix = np.array([
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4]
        ])
        
        # Define the color palette (black, white, blue, red, green)
        colors = [
            [0, 0, 0],       # Black
            [255, 255, 255], # White
            [0, 0, 255],     # Blue
            [255, 0, 0],     # Red
            [0, 255, 0]      # Green
        ]
        
        # Generate the colored image from the matrix
        output_path = 'valid_bw_qr.png'
        colored_image = generate_colored_image_from_matrix(original_matrix, colors, output_path)
        
        # Feed the colored image to the qr_to_color_matrix function
        color_matrix, _ = qr_to_color_matrix(output_path, num_colors=5)
        
        # Print the original matrix
        print("Original Matrix:\n", original_matrix)
        
        # Print the matrix returned by qr_to_color_matrix function
        print("Matrix Returned by qr_to_color_matrix:\n", color_matrix)

if __name__ == '__main__':
    unittest.main()