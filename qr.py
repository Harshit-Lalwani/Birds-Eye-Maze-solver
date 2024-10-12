import networkx as nx
import matplotlib.pyplot as plt

# Function to perform DFS and find connected components
def dfs(matrix, i, j, visited, rows, cols, group):
    stack = [(i, j)]
    component = []  # store all coordinates part of the same component

    # Directions for 8-connected grid (including diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while stack:
        x, y = stack.pop()
        if visited[x][y]:
            continue
        visited[x][y] = True
        component.append((x, y))

        # Check all neighbors
        for dx, dy in directions:
            nx_, ny = x + dx, y + dy
            if (0 <= nx_ < rows and 0 <= ny < cols and
                matrix[nx_][ny] == group and not visited[nx_][ny]):
                stack.append((nx_, ny))

    return component

def build_graph_from_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    visited = [[False for _ in range(cols)] for _ in range(rows)]
    graph = nx.Graph()
    component_nodes = []  # List to store connected components

    # Loop through the matrix to find all connected components
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j]:
                group = matrix[i][j]
                component = dfs(matrix, i, j, visited, rows, cols, group)
                # Create a super node for the component
                super_node_id = f'component-{len(component_nodes)}'
                graph.add_node(super_node_id, pos=(i, j))  # Store position of the super node

                # Add edges between all points in the component
                for (x, y) in component:
                    graph.add_node((x, y), pos=(x, y))  # Add each point as a node with position
                    graph.add_edge(super_node_id, (x, y))  # Connect super node to individual points

                # Store the component node
                component_nodes.append(super_node_id)

    # Create edges between connected components (super nodes)
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:  # If current is a road
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx_, ny = i + dx, j + dy
                    if (0 <= nx_ < rows and 0 <= ny < cols and matrix[nx_][ny] == 0):
                        zero_component_id = f'component-{len(component_nodes) - 1}'  # Assuming last node is the zero component
                        graph.add_edge(f'component-{len(component_nodes) - 1}', (i, j))

    return graph

def draw_graph(graph):
    pos = nx.get_node_attributes(graph, 'pos')  # Use positions stored in graph nodes
    nx.draw(graph, pos, with_labels=True, node_color='blue', node_size=100, edge_color='black')
    plt.show()

# Example matrix (you can replace this with your actual matrix)
matrix = [
    [1, 1, 0, 1, 1],
    [1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0]
]

# Build the graph
graph = build_graph_from_matrix(matrix)
draw_graph(graph)
/home/paarth/flaskapp/esw/WhatsApp Image 2024-10-01 at 15.37.19.jpeg