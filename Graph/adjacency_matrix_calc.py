import os
import re

def calculate_adjacency_matrix(graph, directed):
    """
    graph:
    {
        "A": {"B": 1, "C": 1},
        "B": {"A": 1},
        ...
    }
    """
    # Sort vertices naturally (handles A-D, N1-N45, V1-V8, etc.)
    def natural_sort_key(vertex):
        match = re.search(r'\d+', vertex)
        if match:
            prefix = vertex[:match.start()]
            number = int(match.group())
            return (prefix, number)
        else:
            return (vertex, 0)
    
    vertices = sorted(graph.keys(), key=natural_sort_key)
    n = len(vertices)

    adjacency_matrix = [[0] * n for _ in range(n)]
    vertex_index = {v: i for i, v in enumerate(vertices)}

    for vertex, edges in graph.items():
        i = vertex_index[vertex]
        for neighbor, weight in edges.items():
            j = vertex_index[neighbor]
            adjacency_matrix[i][j] = weight
            if not directed:
                adjacency_matrix[j][i] = weight

    return adjacency_matrix


def read_graph_from_file(file_path, directed=None):
    graph = {}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return graph

    # Check for header format: DIRECTED/UNDIRECTED
    if lines[0] in ("DIRECTED", "UNDIRECTED"):
        is_dir = (lines[0] == "DIRECTED")
        edge_lines = lines[2:]  # Skip header and vertex/edge counts line
    else:
        is_dir = directed
        edge_lines = lines

    for edge_line in edge_lines:
        tokens = edge_line.split()
        if len(tokens) >= 2:
            u, v = tokens[0], tokens[1]
            w = int(tokens[2]) if len(tokens) >= 3 else 1
            if u not in graph:
                graph[u] = {}
            if v not in graph:
                graph[v] = {}
            graph[u][v] = w
            if not is_dir:
                graph[v][u] = w

    return graph


def process_dataset(base_dir, directed):
    erroneous_graphs = []
    graph_type = "directed" if directed else "undirected"

    print(f"\n========== Processing {graph_type} graphs ==========\n")

    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        graph_file = os.path.join(folder_path, f"graph_{folder}.txt")
        print(f"Processing {graph_file}")

        try:
            graph = read_graph_from_file(graph_file, directed)
            adjacency_matrix = calculate_adjacency_matrix(graph, directed)

            output_file = os.path.join(
                folder_path,
                f"adjacency_matrix_{folder}.txt",
            )

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(str(adjacency_matrix))

            print(f"Saved {output_file}")

        except Exception as e:
            erroneous_graphs.append(graph_file)
            print(f"ERROR: {e}")

        print()

    if erroneous_graphs:
        print(f"{graph_type.upper()} ERRORS:")
        for g in erroneous_graphs:
            print(" -", g)
    else:
        print(f"All {graph_type} graphs processed successfully! NO ERRORS!")


def delete_adjacency_matrices(base_dir):
    """
    Deletes all adjacency_matrix_XX.txt files under the given dataset directory.
    """
    if not os.path.exists(base_dir):
        return

    deleted = 0
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        matrix_file = os.path.join(
            folder_path,
            f"adjacency_matrix_{folder}.txt"
        )

        if os.path.exists(matrix_file):
            os.remove(matrix_file)
            print(f"Deleted {matrix_file}")
            deleted += 1

    print(f"Deleted {deleted} adjacency matrix file(s) from {base_dir}.")


def delete_all_adjacency_matrices():
    delete_adjacency_matrices("Graph/graph_samples/undirected")
    delete_adjacency_matrices("Graph/graph_samples/directed")


if __name__ == "__main__":
    # Process undirected graphs
    process_dataset(
        "Graph/graph_samples/undirected",
        directed=False,
    )

    # Process directed graphs
    process_dataset(
        "Graph/graph_samples/directed",
        directed=True,
    )