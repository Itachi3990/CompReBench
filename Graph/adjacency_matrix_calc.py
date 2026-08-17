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
    # Sort vertices naturally (handles both A-D and N1-N45)
    def natural_sort_key(vertex):
        import re
        # Try to extract numeric part for numeric sorting
        match = re.search(r'\d+', vertex)
        if match:
            # Return tuple: (prefix, number) for proper sorting
            # e.g., "N45" -> ("N", 45), "V8" -> ("V", 8)
            prefix = vertex[:match.start()]
            number = int(match.group())
            return (prefix, number)
        else:
            # For non-numeric vertices like A, B, C
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


def read_graph_from_file(file_path, directed):
    graph = {}
    vertex_prefix = None
    vertex_count = None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
        # First pass: extract metadata from the description line
        for line in lines:
            line = line.strip()
            # Look for: "Graph X: a ... graph with N vertices named V1 through VN."
            if "vertices named" in line:
                # Try pattern 1: "with N vertices named V1 through V8"
                m = re.search(r"with (\d+) vertices named (\w+) through (\w+)", line)
                if m:
                    vertex_count = int(m.group(1))
                    start_vertex = m.group(2)
                    end_vertex = m.group(3)
                    
                    # Determine prefix (letters) and number range
                    if start_vertex.isalpha() and end_vertex.isalpha():
                        # All letters: A through D
                        vertex_prefix = ""
                        for i in range(vertex_count):
                            vertex_name = chr(ord(start_vertex) + i)
                            graph[vertex_name] = {}
                    else:
                        # Mixed or numeric: V1 through V8, N1 through N45, Y1 through Y60
                        # Extract the letter prefix
                        prefix_match = re.match(r"([A-Z]*)(\d+)", start_vertex)
                        if prefix_match:
                            vertex_prefix = prefix_match.group(1)
                            for i in range(1, vertex_count + 1):
                                vertex_name = f"{vertex_prefix}{i}"
                                graph[vertex_name] = {}
                    break
                
                # Try pattern 2: "with 5 vertices named A, B, C, D, and E"
                m = re.search(r"with (\d+) vertices named (.+?)(?:\.|$)", line)
                if m and "through" not in line:  # Only if pattern 1 didn't match
                    vertex_count = int(m.group(1))
                    vertices_text = m.group(2)
                    
                    # Parse comma-separated vertices
                    # Handle: "A, B, C, D, and E" or "A and B"
                    vertices_text = vertices_text.replace(", and ", ", ")
                    vertices_text = vertices_text.replace(" and ", ", ")
                    vertex_names = [v.strip() for v in vertices_text.split(",") if v.strip()]
                    
                    if len(vertex_names) == vertex_count:
                        for vertex_name in vertex_names:
                            graph[vertex_name] = {}
                    break
        
        # Second pass: parse edges
        for line_raw in lines:
            line = line_raw.strip()

            if not line.startswith("-"):
                continue

            if directed:
                # Handle: "- A has no outgoing edges"
                if "no outgoing edges" in line:
                    m = re.match(r"-\s*(\w+)\s+has no outgoing edges", line)
                    if m:
                        # Just mark that we've seen this vertex (it's already initialized)
                        pass
                    continue

                # Try pattern 1: "- A has outgoing edges to B and C, each with weight 1."
                #           or: "- B has an outgoing edge to D, each with weight 4."
                m = re.match(
                    r"-\s*(\w+)\s+has (an )?outgoing edge[s]?\s+to\s+(.*),\s*each with weight\s+(\d+)\.",
                    line,
                )

                if m:
                    vertex = m.group(1)
                    neighbors_text = m.group(3)  # Group 2 is the optional "an ", so skip it
                    weight = int(m.group(4))

                    neighbors_text = neighbors_text.replace(", and ", ", ")
                    neighbors_text = neighbors_text.replace(" and ", ", ")

                    neighbors = [
                        x.strip()
                        for x in neighbors_text.split(",")
                        if x.strip()
                    ]

                    if vertex in graph:
                        for neighbor in neighbors:
                            if neighbor in graph:
                                graph[vertex][neighbor] = weight

                    continue

                # Try pattern 2: "- N1 has outgoing edges to N7, N18, and N34, with weights 7, 842, and 19 respectively."
                #           or: "- A has outgoing edges to B and C, with weights 6 and 2 respectively."
                #           or: "- B has an outgoing edge to D, with weight 5 respectively." (edge case)
                m = re.match(
                    r"-\s*(\w+)\s+has (an )?outgoing edge[s]?\s+to\s+(.*),\s*with weights?\s+(.*)\s+respectively\.",
                    line,
                )

                if m:
                    vertex = m.group(1)
                    neighbors_text = m.group(3)  # Group 2 is the optional "an ", so skip it
                    weights_text = m.group(4)

                    # Parse neighbors
                    neighbors_text = neighbors_text.replace(", and ", ", ")
                    neighbors_text = neighbors_text.replace(" and ", ", ")
                    neighbors = [
                        x.strip()
                        for x in neighbors_text.split(",")
                        if x.strip()
                    ]

                    # Parse weights - handle both "," and " and " separators
                    weights_text = weights_text.replace(", and ", ", ")
                    weights_text = weights_text.replace(" and ", ", ")
                    weights = [int(x.strip()) for x in weights_text.split(",")]

                    # Match neighbors to weights
                    if len(neighbors) == len(weights):
                        if vertex in graph:
                            for neighbor, weight in zip(neighbors, weights):
                                if neighbor in graph:
                                    graph[vertex][neighbor] = weight
                    else:
                        raise ValueError(
                            f"Mismatch: {len(neighbors)} neighbors but {len(weights)} weights in line: {line}"
                        )

                    continue

            else:
                # Undirected graph patterns

                # Try pattern 1: "- A is incident to B and C, each with weight 1."
                #           or: "- Y1 is incident to Y2 and Y3, each with weight 1."
                m = re.match(
                    r"-\s*(\w+)\s+is incident to\s+(.*),\s*each with weight\s+(\d+)\.",
                    line,
                )

                if m:
                    vertex = m.group(1)
                    neighbors_text = m.group(2)
                    weight = int(m.group(3))

                    neighbors_text = neighbors_text.replace(", and ", ", ")
                    neighbors_text = neighbors_text.replace(" and ", ", ")

                    neighbors = [
                        x.strip()
                        for x in neighbors_text.split(",")
                        if x.strip()
                    ]

                    if vertex in graph:
                        for neighbor in neighbors:
                            if neighbor in graph:
                                graph[vertex][neighbor] = weight
                                graph[neighbor][vertex] = weight

                    continue

                # Try pattern 2: "- V1 is incident to V2, V3, V4, with weights 1, 2, 3 respectively."
                #           or: "- Y1 is incident to Y2, Y3, with weights 1, 1 respectively."
                #           or: "- Y58 is incident to Y55, with weights 1 respectively."
                m = re.match(
                    r"-\s*(\w+)\s+is incident to\s+(.*),\s*with weights?\s+(.*)\s+respectively\.",
                    line,
                )

                if m:
                    vertex = m.group(1)
                    neighbors_text = m.group(2)
                    weights_text = m.group(3)

                    # Parse neighbors
                    neighbors_text = neighbors_text.replace(", and ", ", ")
                    neighbors_text = neighbors_text.replace(" and ", ", ")
                    neighbors = [
                        x.strip()
                        for x in neighbors_text.split(",")
                        if x.strip()
                    ]

                    # Parse weights - handle both "," and " and " separators
                    weights_text = weights_text.replace(", and ", ", ")
                    weights_text = weights_text.replace(" and ", ", ")
                    weights = [int(x.strip()) for x in weights_text.split(",")]

                    # Match neighbors to weights
                    if len(neighbors) == len(weights):
                        if vertex in graph:
                            for neighbor, weight in zip(neighbors, weights):
                                if neighbor in graph:
                                    graph[vertex][neighbor] = weight
                                    graph[neighbor][vertex] = weight
                    else:
                        raise ValueError(
                            f"Mismatch: {len(neighbors)} neighbors but {len(weights)} weights in line: {line}"
                        )

                    continue

    return graph


def process_dataset(base_dir, directed):
    erroneous_graphs = []

    graph_type = "directed" if directed else "undirected"

    print(f"\n========== Processing {graph_type} graphs ==========\n")

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


# ==========================
# Main
# ==========================

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

# If needed later, remove every generated adjacency matrix from both the directed and undirected graph datasets.
# delete_all_adjacency_matrices()