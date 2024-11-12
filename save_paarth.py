import cv2
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx

color_to_number = {
    'black': 1,
    'white': 0,
    '-1': -1  # Undefined areas
}


def draw_graph_with_path_on_image(image, path, matrix):
    # Resize image to fit grid matrix dimensions if needed
    cell_size = image.shape[0] // len(matrix)  # Adjust based on matrix size and image dimensions

    # Draw a line between each consecutive point in the path
    for idx in range(len(path) - 1):
        (i1, j1), (i2, j2) = path[idx], path[idx + 1]
        
        # Calculate the center of each cell
        start_point = (j1 * cell_size + cell_size // 2, i1 * cell_size + cell_size // 2)
        end_point = (j2 * cell_size + cell_size // 2, i2 * cell_size + cell_size // 2)
        
        # Draw line between points
        cv2.line(image, start_point, end_point, (0, 0, 255), 5)  # Red color with thickness 2

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Shortest Path Highlighted on Image")
    plt.show()


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

def matrix_to_graph(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    print(rows)
    print(cols)
    G = nx.Graph()
    row_ends = {}
    col_ends = {}

    # Create nodes and edges for rows
    for i in range(rows):
        start = -1
        for j in range(cols):
            # print(i,j)
            if matrix[i][j] == 1:
                if start == -1:
                    start = j
            else:
                if start != -1:
                    end = j - 1
                    G.add_node((i, start))
                    G.add_node((i, end))
                    G.add_edge((i, start), (i, end), weight=end - start + 1)
                    row_ends[(i, end)] = (i, start)
                    start = -1
        if start != -1:  # Handle the case where row ends with 1s
            end = cols - 1
            G.add_node((i, start))
            G.add_node((i, end))
            G.add_edge((i, start), (i, end), weight=end - start + 1)
            row_ends[(i, end)] = (i, start)

    # Create nodes and edges for columns
    for j in range(cols):
        start = -1
        for i in range(rows):
            if matrix[i][j] == 1:
                if start == -1:
                    start = i
            else:
                if start != -1:
                    end = i - 1
                    G.add_node((start, j))
                    G.add_node((end, j))
                    G.add_edge((start, j), (end, j), weight=end - start + 1)
                    col_ends[(end, j)] = (start, j)
                    start = -1
        if start != -1:  # Handle the case where column ends with 1s
            end = rows - 1
            G.add_node((start, j))
            G.add_node((end, j))
            G.add_edge((start, j), (end, j), weight=end - start + 1)
            col_ends[(end, j)] = (start, j)

    # Connect row and column ends
    for (i, j), _ in row_ends.items():
        if (i, j) in col_ends:
            G.add_edge(row_ends[(i, j)], col_ends[(i, j)], weight=1)

    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def add_edges_to_graph(G, matrix):
    def has_continuous_row(i, start, end):
        return all(matrix[i][k] == 1 for k in range(start, end + 1))

    def has_continuous_col(j, start, end):
        return all(matrix[k][j] == 1 for k in range(start, end + 1))

    existing_vertices = list(G.nodes)
    for v1 in existing_vertices:
        for v2 in existing_vertices:
            if v1 != v2 and not G.has_edge(v1, v2):
                i1, j1 = v1
                i2, j2 = v2
                if i1 == i2:
                    if has_continuous_row(i1, min(j1, j2), max(j1, j2)):
                        G.add_edge(v1, v2, weight=abs(j1 - j2))
                elif j1 == j2:
                    if has_continuous_col(j1, min(i1, i2), max(i1, i2)):
                        G.add_edge(v1, v2, weight=abs(i1 - i2))
    return G

def draw_graph_with_shortest_path(G, shortest_path,matrix):
    pos = {(i, j): (j, -i) for i in range(len(matrix)) for j in range(len(matrix[0]))}
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=10, font_weight='bold', edge_color='gray')
    if shortest_path:
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)
    plt.title("Graph with Shortest Path Highlighted in Red")
    plt.show()

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
            G = matrix_to_graph(numeric_matrix)

# Optionally, add edges based on additional conditions
            G = add_edges_to_graph(G, numeric_matrix)

            # Define start and end nodes based on existing graph nodes
            start_node = (1,1)  # Change as necessary
            end_node = (23,23)    # Change as necessary

            # Ensure start and end nodes exist in the graph
            if start_node in G.nodes and end_node in G.nodes:
                # Find the shortest path
                shortest_path = nx.shortest_path(G, source=start_node, target=end_node)
                
                # Draw the graph and highlight the shortest path
                draw_graph_with_shortest_path(G, shortest_path,numeric_matrix)
                print("Shortest path from", start_node, "to", end_node, ":", shortest_path)
                draw_graph_with_path_on_image(square_image, shortest_path, numeric_matrix)
            else:
                print("Either the start or end node does not exist in the graph.")


if __name__ == "__main__":
    main()
