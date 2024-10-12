import cv2
import numpy as np
from sklearn.cluster import KMeans

def qr_to_color_matrix(image_path, num_colors=5, target_size=(15, 15)):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Unable to load image at path: {image_path}")
    
    # Convert the image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Apply k-means clustering to segment the image into num_colors clusters
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    
    # Get the cluster centers (the colors)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Assign each pixel to the nearest cluster center
    labels = kmeans.labels_
    
    # Calculate the dimensions of the original image
    height, width = image_rgb.shape[:2]
    
    # Calculate the block size to fit into a 15x15 matrix
    block_height = height // target_size[0]
    block_width = width // target_size[1]

    # Create the color matrix with the target size
    color_matrix = np.zeros(target_size, dtype=int)
    
    # Create a dictionary to store the color mappings
    color_mapping = {}
    
    # Ensure that black (near [0, 0, 0]) is assigned the value 1
    next_color_code = 2  # Start assigning numbers from 2 for non-black colors

    # Function to determine if a color is black
    # Function to determine if a color is black
def is_black(rgb_color):
    threshold = 30  # Set a threshold for what is considered black
    return np.all(np.array(rgb_color) < threshold)  # Convert the list to a NumPy array for element-wise comparison


    for i in range(target_size[0]):
        for j in range(target_size[1]):
            # Calculate the indices for the current block
            start_row = i * block_height
            end_row = (i + 1) * block_height
            start_col = j * block_width
            end_col = (j + 1) * block_width
            
            # Extract the labels for the current block
            block_labels = labels[start_row * width + start_col:end_row * width + end_col]
            block_average = np.bincount(block_labels).argmax()
            
            # Get the RGB color of the current block from the colors array
            block_color_rgb = list(colors[block_average])
            
            # Assign 1 for black, and other unique numbers for other colors
            if is_black(block_color_rgb):
                color_matrix[i, j] = 1
            else:
                # Check if the color is already mapped, if not, assign a new number
                color_tuple = tuple(block_color_rgb)
                if color_tuple not in color_mapping:
                    color_mapping[color_tuple] = next_color_code
                    next_color_code += 1
                color_matrix[i, j] = color_mapping[color_tuple]

    return color_matrix, colors, color_mapping

# Example usage
if __name__ == '__main__':

    image_path = '/home/paarth/flaskapp/esw/WhatsApp Image 2024-10-01 at 15.37.19.jpeg'  # Adjust the path as necessary
    try:
        color_matrix, colors, color_mapping = qr_to_color_matrix(image_path, num_colors=5, target_size=(15, 15))
        np.set_printoptions(threshold=np.inf)
        print("Color Matrix:\n", color_matrix)
        print("Colors:\n", colors)
        print("Color Mappings (non-black):\n", color_mapping)
    except FileNotFoundError as e:
        print(e)
