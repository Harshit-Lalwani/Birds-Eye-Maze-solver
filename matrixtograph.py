import networkx as nx
import matplotlib.pyplot as plt

def matrix_to_graph(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    G = nx.Graph()
    row_ends = {}
    col_ends = {}

    for i in range(rows):
        start = -1
        for j in range(cols):
            if matrix[i][j] == 1:
                if start == -1:
                    start = j
            else:
                if start != -1:
                    end = j - 1
                    if start != end:
                        G.add_node((i, start))
                        G.add_node((i, end))
                        G.add_edge((i, start), (i, end), weight=end - start + 1)
                    row_ends[(i, end)] = (i, start)
                    start = -1
        if start != -1:
            end = cols - 1
            if start != end:
                G.add_node((i, start))
                G.add_node((i, end))
                G.add_edge((i, start), (i, end), weight=end - start + 1)
            row_ends[(i, end)] = (i, start)

    for j in range(cols):
        start = -1
        for i in range(rows):
            if matrix[i][j] == 1:
                if start == -1:
                    start = i
            else:
                if start != -1:
                    end = i - 1
                    if start != end:
                        G.add_node((start, j))
                        G.add_node((end, j))
                        G.add_edge((start, j), (end, j), weight=end - start + 1)
                    col_ends[(end, j)] = (start, j)
                    start = -1
        if start != -1:
            end = rows - 1
            if start != end:
                G.add_node((start, j))
                G.add_node((end, j))
                G.add_edge((start, j), (end, j), weight=end - start + 1)
            col_ends[(end, j)] = (start, j)

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

def draw_graph_with_shortest_path(G, shortest_path):
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

matrix = [
    [1, 1, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1]
]

G = matrix_to_graph(matrix)
G = add_edges_to_graph(G, matrix)

start_vertex = (0, 0)
end_vertex = (3, 4)

shortest_path = nx.dijkstra_path(G, start_vertex, end_vertex)

print("Shortest path from", start_vertex, "to", end_vertex, ":", shortest_path)

draw_graph_with_shortest_path(G, shortest_path)
