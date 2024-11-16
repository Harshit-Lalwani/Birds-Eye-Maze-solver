import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def read_matrix_from_file(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        row = []
        num = ''
        for char in file.read():
            if char.isdigit() or char == '-':
                num += char
            elif num:
                row.append(int(num))
                num = ''
            if char == '\n' and row:
                matrix.append(row)
                row = []
        if row:
            matrix.append(row)
    return matrix

def is_valid_region(matrix, i, j):
    # Check if the 3x3 region starting at (i, j) is valid (all cells are 1)
    for x in range(i, i + 2):
        for y in range(j, j + 2):
            if x >= len(matrix) or y >= len(matrix[0]) or matrix[x][y] != 1:
                return False
    return True

def matrix_to_graph(matrix):
    graph = nx.Graph()
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0

    # Add edges between valid 3x3 regions
    for i in range(n - 2):
        for j in range(m - 2):
            if is_valid_region(matrix, i, j):
                if is_valid_region(matrix, i + 1, j):
                    graph.add_edge((i, j), (i + 1, j), weight=1)
                if is_valid_region(matrix, i, j + 1):
                    graph.add_edge((i, j), (i, j + 1), weight=1)

    return graph

def draw_grid(matrix, graph, shortest_path, file_path):
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    grid = np.zeros((n, m, 3), dtype=int)

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 1:
                grid[i, j] = [0, 0, 139]  # Dark blue
            elif matrix[i][j] == -1:
                grid[i, j] = [255, 0, 0]  # Red
            else:
                grid[i, j] = [173, 216, 230]  # Light blue

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, interpolation='nearest')

    for (u, v, d) in graph.edges(data=True):
        x1, y1 = u
        x2, y2 = v
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        plt.text(mid_y, mid_x, f"{d['weight']}", color='white', ha='center', va='center')
        plt.plot([y1, y2], [x1, x2], color='blue', linewidth=2)

    for (u, v) in shortest_path:
        x1, y1 = u
        x2, y2 = v
        plt.plot([y1, y2], [x1, x2], color='green', linewidth=20)

    plt.xticks(ticks=range(m), labels=range(m))
    plt.yticks(ticks=range(n), labels=range(n))
    plt.grid(True)
    plt.savefig(file_path)

def print_edges(graph):
    print("Edges with their weight and coordinates:")
    for (u, v, d) in graph.edges(data=True):
        weight = d['weight']
        print(f"Edge from {u} to {v}: weight = {weight}")

if __name__ == "__main__":
    matrix = read_matrix_from_file('ESW/output.txt')
    graph = matrix_to_graph(matrix)
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    start_node = (1, 1)
    end_node = (21, 21)
    shortest_path_edges = []
    if start_node in graph and end_node in graph:
        try:
            shortest_path = nx.shortest_path(graph, source=start_node, target=end_node, weight='weight')
            print(f"Shortest path: {shortest_path}")
            shortest_path_edges = list(zip(shortest_path, shortest_path[1:]))
        except nx.NetworkXNoPath:
            print("No valid path found.")
    else:
        print("Start or end node not in graph.")
    draw_grid(matrix, graph, shortest_path_edges, 'graph.png')
    # print_edges(graph)