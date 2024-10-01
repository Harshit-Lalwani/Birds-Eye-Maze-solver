import cv2
import numpy as np
from sklearn.cluster import KMeans

def qr_to_color_matrix(image_path, num_colors=2):
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
    
    # Reshape the labels back to the original image shape
    color_matrix = labels.reshape(image_rgb.shape[:2])
    
    return color_matrix, colors

# Example usage
if __name__ == '__main__':
    image_path = 'path_to_qr_code_image.png'
    try:
        color_matrix, colors = qr_to_color_matrix(image_path)
        print("Color Matrix:\n", color_matrix)
        print("Colors:\n", colors)
    except FileNotFoundError as e:
        print(e)