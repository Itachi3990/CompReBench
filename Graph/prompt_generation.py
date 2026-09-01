import os
import random
import re
import ast
import json
from pathlib import Path
import sys
from collections import defaultdict

from graph import Graph

# Fix stdout encoding for printing emojis on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Use os.urandom for true randomness via SystemRandom
random = random.SystemRandom()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
USE_SLM = True  # Set to True to use local Gemma SLM (via LM Studio), False to use Gemini via agy CLI
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_MODEL = "google/gemma-3-4b"
SLM_TEMPERATURE = 0.2

# Set to None to process all combinations, or an integer to limit prompts
MAX_PROMPTS_COUNT = 3  # Change to 4, 10, etc. to limit
PROBLEMS_FROM_EACH_CATEGORY = 5
NUMBER_OF_GRAPHS = 10
# Bias towards selecting last graph folder (tunable)
# 0.0 = no bias, 0.5 = last folder has 50% chance, 0.9 = last folder has 90% chance
LAST_GRAPH_BIAS = 0.01

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_BASE = os.path.join(BASE_DIR, "Problems")
DIRECTED_GRAPHS_BASE = os.path.join(BASE_DIR, "graph_samples", "directed")
UNDIRECTED_GRAPHS_BASE = os.path.join(BASE_DIR, "graph_samples", "undirected")
LLM_LOG_FILE = os.path.join(BASE_DIR, "llm_log.txt")
PROGRESS_FILE = os.path.join(BASE_DIR, "prompt_generation_progress.json")

# ============================================================================
# LLM / SLM INFERENCE HELPER
# ============================================================================

def call_model(prompt: str, max_tokens: int = 2048) -> str:
    """
    Unified model invocation function.
    If USE_SLM is True, runs inference locally using Gemma via LM Studio API.
    If USE_SLM is False, calls Gemini via the agy CLI.
    """
    if USE_SLM:
        import requests
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": SLM_TEMPERATURE,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        import subprocess
        result = subprocess.run(
            [
                r"C:\Users\ASUS\AppData\Local\agy\bin\agy.exe",
                "--model",
                "gemini-3.7-flash-high",
                "-p",
                prompt,
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

# Algorithm reference dictionary
GRAPH_ALGORITHMS = {
    "All_Pairs_Shortest_Path": [
        "Floyd-Warshall", "Johnson's Algorithm", "Repeated Dijkstra", "Repeated Bellman-Ford",
    ],
    "BFS_DFS": [
        "BFS", "Breadth First Search", "DFS", "Depth First Search",
        "Breadth-First Search", "Depth-First Search", "Graph Traversal", "Level Order Traversal",
    ],
    "Bipartite_Matching": [
        "Hopcroft-Karp", "Hungarian Algorithm", "Munkres",
        "Ford-Fulkerson", "Dinic", "Blossom Algorithm",
        "Kuhn's Algorithm", "Maximum Matching",
    ],
    "Cycle_Check": [
        "DFS", "Depth First Search", "Union-Find", "Kahn's Algorithm",
        "Tarjan's SCC", "Kosaraju's Algorithm", "Topological Sort",
    ],
    "MST": [
        "Kruskal's Algorithm", "Prim's Algorithm", "Boruvka's Algorithm", "Reverse-Delete Algorithm",
    ],
    "Max_Flow": [
        "Ford-Fulkerson", "Edmonds-Karp", "Dinic", "Push-Relabel",
        "Relabel-to-Front", "Boykov-Kolmogorov", "Capacity Scaling",
    ],
    "Minimum_Cut": [
        "Karger's Algorithm", "Karger-Stein", "Stoer-Wagner", "Gomory-Hu",
        "Ford-Fulkerson", "Edmonds-Karp", "Dinic", "Max-Flow Min-Cut",
    ],
    "SCC": [
        "Tarjan's SCC", "Tarjan's Algorithm", "Kosaraju's Algorithm",
        "Kosaraju-Sharir", "Kosaraju", "Path-based strong component algorithm", "Gabow's Algorithm",
    ],
    "Shortest_Path": [
        "Dijkstra", "Bellman-Ford", "Floyd-Warshall", "A*",
        "Johnson's Algorithm", "BFS", "Bidirectional Dijkstra",
        "SPFA", "D* Lite",
    ],
    "Topological_Sort": [
        "Topological Sort", "Kahn's Algorithm", "DFS", "Depth First Search",
        "Tarjan's Topological Sort", "Topological Ordering",
    ],
    "TSP": [
        "Held-Karp", "Branch and Bound", "Nearest Neighbor",
        "Christofides", "Simulated Annealing", "Genetic Algorithm",
        "Ant Colony Optimization", "Lin-Kernighan", "LKH", "Concorde",
        "Dynamic Programming",
    ],
}

# Prompt template (modular, easily replaceable)
GRAPH_PROMPT = """DO NOT USE THE INTERNET.

Consider the graph described below:

{graph_description}

After that, consider the following graph problem:

{problem_description}

Decide which algorithm(s) you would use to solve this problem for the given graph. Do NOT solve the problem itself.

Then provide the exact graph metadata and algorithm answer below. Use deterministic formatting and do not add extra labels or commentary.

Your output must EXACTLY follow this format, in this exact order:

ADJACENCY MATRIX: [[0, 1, 0], [0, 0, 3], [2, 0, 0]]
IS_DIRECTED: True
NUMBER_OF_VERTICES: 3
NUMBER_OF_EDGES: 3
HAS_SELF_LOOPS: False
HAS_PARALLEL_EDGES: False
EDGE_LIST: [["A", "B", 1], ["B", "C", 3], ["C", "A", 2]]
VERTEX_DEGREES: {{"A": {{"in_degree": 1, "out_degree": 1}}, "B": {{"in_degree": 1, "out_degree": 1}}, "C": {{"in_degree": 1, "out_degree": 1}}}}
ALGORITHM NAME: <ALGORITHM NAME>
EXPLANATION OF ALGORITHM CHOICE: <EXPLANATION>

Important:
- Use Python literal formatting for all structured values.
- Boolean values must be written as True or False.
- EDGE_LIST must be a list of [source, destination, weight] triples.
- VERTEX_DEGREES must map each vertex to a dictionary with in_degree and out_degree keys.
- For undirected graphs, in_degree and out_degree should be equal to the undirected degree.
- Sort the EDGE_LIST and VERTEX_DEGREES entries in a deterministic vertex order before writing them.
- The fields ALGORITHM NAME and EXPLANATION OF ALGORITHM CHOICE are mandatory and must remain in the output. Whatever algorithm(s) you choose, clearly state the algorithm name(s) and provide a brief explanation (10-30 words) of why that algorithm is appropriate for this graph and problem.
"""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_file(filepath):
    """Load content from a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

import re

def _natural_vertex_key(vertex):
    """
    Natural sort key for vertex labels.

    Examples:
        A, B, C              -> A, B, C
        V1, V2, V10          -> V1, V2, V10
        A1, A2, A10          -> A1, A2, A10
        node1, node2, node10 -> node1, node2, node10

    Non-numeric parts are compared case-insensitively.
    """

    parts = re.split(r'(\d+)', vertex)

    return tuple(
        (0, int(part)) if part.isdigit()
        else (1, part.lower())
        for part in parts
        if part
    )


def graph_to_description(graph):
    """Build the natural-language graph description directly from a Graph object."""
    if not isinstance(graph, Graph):
        raise TypeError("graph_to_description expects a Graph instance.")

    vertices = sorted(graph.vertices, key=_natural_vertex_key)
    graph_type = "DIRECTED" if graph.is_directed() else "UNDIRECTED"
    vertex_word = "vertex" if len(vertices) == 1 else "vertices"

    description = (
        f"Graph: a {graph_type.lower()} weighted graph with {len(vertices)} {vertex_word}. "
        f"The vertices are {', '.join(vertices)}.\n"
    )
    description += f"The graph contains {graph.get_number_of_edges()} {'edge' if graph.get_number_of_edges() == 1 else 'edges'}.\n"
    description += "Edge information:\n"

    if graph.is_directed():
        adjacency = graph.adjacency
        for vertex in vertices:
            outgoing = sorted(adjacency.get(vertex, {}).items(), key=lambda item: _natural_vertex_key(item[0]))
            if not outgoing:
                description += f"- Vertex {vertex} has no outgoing edges.\n"
                continue

            edge_descriptions = [f"{neighbor} with weight {weight}" for neighbor, weight in outgoing]
            if len(edge_descriptions) == 1:
                description += f"- Vertex {vertex} has an outgoing edge to {edge_descriptions[0]}.\n"
            else:
                edge_text = ", ".join(edge_descriptions[:-1]) + f", and {edge_descriptions[-1]}"
                description += f"- Vertex {vertex} has outgoing edges to {edge_text}.\n"
    else:
        seen = set()
        neighbors = {vertex: [] for vertex in vertices}
        for u, v, weight in graph.get_edge_list():
            pair = tuple(sorted((u, v)))
            if pair in seen:
                continue
            seen.add(pair)
            neighbors[u].append((v, weight))
            neighbors[v].append((u, weight))

        for vertex in vertices:
            incident = sorted(neighbors.get(vertex, []), key=lambda item: _natural_vertex_key(item[0]))
            if not incident:
                description += f"- Vertex {vertex} is not connected to any other vertex.\n"
                continue

            edge_descriptions = [f"{neighbor} with weight {weight}" for neighbor, weight in incident]
            if len(edge_descriptions) == 1:
                description += f"- Vertex {vertex} is connected to {edge_descriptions[0]}.\n"
            else:
                edge_text = ", ".join(edge_descriptions[:-1]) + f", and {edge_descriptions[-1]}"
                description += f"- Vertex {vertex} is connected to {edge_text}.\n"

    return description


def canonicalize_edge_list(edge_list):
    """Normalize edge list output into a deterministic sorted list of triples."""
    normalized = []
    for u, v, weight in edge_list:
        normalized.append((str(u), str(v), int(weight)))
    return sorted(
        normalized,
        key=lambda item: (_natural_vertex_key(item[0]), _natural_vertex_key(item[1]), item[2]),
    )


def canonicalize_vertex_degrees(degrees):
    """Normalize degree output into a deterministic dict keyed by sorted vertices."""
    normalized = {}
    for vertex, metric in degrees.items():
        normalized[str(vertex)] = {
            "in_degree": int(metric.get("in_degree", 0)),
            "out_degree": int(metric.get("out_degree", 0)),
        }
    return {vertex: normalized[vertex] for vertex in sorted(normalized, key=_natural_vertex_key)}


def graph_to_expected_metadata(graph):
    """Return canonical metadata from the Graph object that the LLM is expected to emit."""
    if not isinstance(graph, Graph):
        raise TypeError("graph_to_expected_metadata expects a Graph instance.")

    return {
        "is_directed": bool(graph.is_directed()),
        "number_of_vertices": int(graph.get_number_of_vertices()),
        "number_of_edges": int(graph.get_number_of_edges()),
        "has_self_loops": bool(graph.has_self_loops()),
        "has_parallel_edges": bool(graph.has_parallel_edges()),
        "edge_list": canonicalize_edge_list(graph.get_edge_list()),
        "vertex_degrees": canonicalize_vertex_degrees(graph.get_indegree_and_outdegree_of_every_vertex()),
    }


def graph_to_expected_metadata_string(graph):
    """Return the exact deterministic metadata block expected from the LLM."""
    metadata = graph_to_expected_metadata(graph)
    edge_list_str = str(metadata["edge_list"]).replace("'", '"')
    vertex_degrees_str = str(metadata["vertex_degrees"]).replace("'", '"')
    return (
        f"IS_DIRECTED: {str(metadata['is_directed'])}\n"
        f"NUMBER_OF_VERTICES: {metadata['number_of_vertices']}\n"
        f"NUMBER_OF_EDGES: {metadata['number_of_edges']}\n"
        f"HAS_SELF_LOOPS: {str(metadata['has_self_loops'])}\n"
        f"HAS_PARALLEL_EDGES: {str(metadata['has_parallel_edges'])}\n"
        f"EDGE_LIST: {edge_list_str}\n"
        f"VERTEX_DEGREES: {vertex_degrees_str}\n"
    )


def graph_format_to_description(graph_str):
    """Convert a structured graph representation into a deterministic,
    lossless natural-language description suitable for LLM benchmarking."""

    # ============================================================
    # Parse input
    # ============================================================

    lines = [
        line.strip()
        for line in graph_str.strip().splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        raise ValueError(
            "Graph input must contain at least two lines."
        )

    # ============================================================
    # Graph type
    # ============================================================

    graph_type = lines[0].upper()

    if graph_type not in {"DIRECTED", "UNDIRECTED"}:
        raise ValueError(
            f"Invalid graph type '{lines[0]}'. "
            "Expected DIRECTED or UNDIRECTED."
        )

    # ============================================================
    # Vertex and edge counts
    # ============================================================

    header = lines[1].split()

    if len(header) != 2:
        raise ValueError(
            "Second line must contain exactly two integers: "
            "<vertex_count> <edge_count>."
        )

    try:
        vertex_count, edge_count = map(int, header)
    except ValueError:
        raise ValueError(
            "Vertex count and edge count must be integers."
        )

    if vertex_count < 0:
        raise ValueError("Vertex count cannot be negative.")

    if edge_count < 0:
        raise ValueError("Edge count cannot be negative.")

    # ============================================================
    # Parse edges
    # ============================================================

    edge_lines = lines[2:]

    if len(edge_lines) != edge_count:
        raise ValueError(
            f"Declared {edge_count} edges, "
            f"but found {len(edge_lines)} edge lines."
        )

    edges = []
    all_vertices = set()

    for line_number, line in enumerate(edge_lines, start=3):

        parts = line.split()

        if len(parts) != 3:
            raise ValueError(
                f"Invalid edge on line {line_number}: '{line}'. "
                "Expected: <source> <destination> <weight>."
            )

        source, destination, weight_str = parts

        try:
            weight = int(weight_str)
        except ValueError:
            raise ValueError(
                f"Invalid edge weight on line {line_number}: "
                f"'{weight_str}'. Weight must be an integer."
            )

        edges.append((source, destination, weight))

        all_vertices.add(source)
        all_vertices.add(destination)

    # ============================================================
    # Canonical vertex ordering
    # ============================================================

    # IMPORTANT:
    # Do NOT preserve first-seen order.
    #
    # This gives:
    #   A, B, C, D
    #
    # and:
    #   V1, V2, V3, ..., V9, V10, V11
    #
    # rather than:
    #   V1, V10, V11, V2, V3, ...

    vertices = sorted(
        all_vertices,
        key=_natural_vertex_key
    )

    # ============================================================
    # Validate vertex count
    # ============================================================

    if len(vertices) != vertex_count:
        raise ValueError(
            f"Declared {vertex_count} vertices, "
            f"but found {len(vertices)} unique vertices: "
            f"{vertices}"
        )

    # ============================================================
    # Validate graph structure
    # ============================================================

    # Simple graph = no self-loops.
    for source, destination, weight in edges:

        if source == destination:
            raise ValueError(
                f"Self-loop detected: {source} -> {destination}. "
                "This function expects a simple graph."
            )

    # Check duplicate edges.
    seen_edges = set()

    for source, destination, weight in edges:

        if graph_type == "DIRECTED":

            edge_key = (source, destination)

        else:

            # Treat P-Q and Q-P as the same undirected edge.
            edge_key = frozenset((source, destination))

        if edge_key in seen_edges:
            raise ValueError(
                f"Duplicate edge detected involving "
                f"{source} and {destination}."
            )

        seen_edges.add(edge_key)

    # ============================================================
    # Group edges by source
    # ============================================================

    edges_by_source = {
        vertex: []
        for vertex in vertices
    }

    for source, destination, weight in edges:

        edges_by_source[source].append(
            (destination, weight)
        )

    # IMPORTANT:
    # Sort outgoing edges using the same natural vertex ordering.
    for vertex in edges_by_source:

        edges_by_source[vertex].sort(
            key=lambda x: _natural_vertex_key(x[0])
        )

    # ============================================================
    # Build description
    # ============================================================

    vertex_list = ", ".join(vertices)

    vertex_word = (
        "vertex"
        if vertex_count == 1
        else "vertices"
    )

    description = (
        f"Graph: a {graph_type.lower()} simple weighted graph "
        f"with {vertex_count} {vertex_word}. "
        f"The vertices are {vertex_list}.\n"
    )

    description += (
        f"The graph contains {edge_count} "
        f"{'edge' if edge_count == 1 else 'edges'}.\n"
    )

    description += "Edge information:\n"

    # ============================================================
    # Directed graph
    # ============================================================

    if graph_type == "DIRECTED":

        for vertex in vertices:

            outgoing = edges_by_source[vertex]

            if not outgoing:

                description += (
                    f"- Vertex {vertex} has no outgoing edges.\n"
                )

                continue

            edge_descriptions = [
                f"{destination} with weight {weight}"
                for destination, weight in outgoing
            ]

            if len(edge_descriptions) == 1:

                description += (
                    f"- Vertex {vertex} has an outgoing edge "
                    f"to {edge_descriptions[0]}.\n"
                )

            else:

                edge_text = ", ".join(
                    edge_descriptions[:-1]
                )

                edge_text += (
                    f", and {edge_descriptions[-1]}"
                )

                description += (
                    f"- Vertex {vertex} has outgoing edges "
                    f"to {edge_text}.\n"
                )

    # ============================================================
    # Undirected graph
    # ============================================================

    else:

        # Construct adjacency information in both directions.
        neighbors = {
            vertex: []
            for vertex in vertices
        }

        for source, destination, weight in edges:

            neighbors[source].append(
                (destination, weight)
            )

            neighbors[destination].append(
                (source, weight)
            )

        # Natural-sort every neighbor list.
        for vertex in neighbors:

            neighbors[vertex].sort(
                key=lambda x: _natural_vertex_key(x[0])
            )

        for vertex in vertices:

            incident = neighbors[vertex]

            if not incident:

                description += (
                    f"- Vertex {vertex} is not connected to "
                    f"any other vertex.\n"
                )

                continue

            edge_descriptions = [
                f"{neighbor} with weight {weight}"
                for neighbor, weight in incident
            ]

            if len(edge_descriptions) == 1:

                description += (
                    f"- Vertex {vertex} is connected to "
                    f"{edge_descriptions[0]}.\n"
                )

            else:

                edge_text = ", ".join(
                    edge_descriptions[:-1]
                )

                edge_text += (
                    f", and {edge_descriptions[-1]}"
                )

                description += (
                    f"- Vertex {vertex} is connected to "
                    f"{edge_text}.\n"
                )

    return description

def save_to_log(message):
    """Append message to LLM log file."""
    with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def clear_log():
    """Clear the LLM log file at start."""
    with open(LLM_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

def select_graphs_with_bias(graph_folders, count, bias=None):
    """
    Select 'count' graph folders with optional bias toward the last folder(s).

    Args:
        graph_folders: List of folder names (sorted, e.g., ['01', '02', ..., 'N'])
        count: Number of folders to select
        bias: If None, sample uniformly from all folders including the last one.
              If a float in [0.0, 1.0], bias the selection toward the final two folders.

    Returns:
        List of selected folder names
    """
    if not graph_folders:
        return []

    if bias is None:
        return [random.choice(graph_folders) for _ in range(count)]

    selected = []
    for _ in range(count):
        if random.random() < bias and len(graph_folders) > 2:
            # Select from the last two folders with probability 'bias'
            available = graph_folders[-2:]
            selected.append(random.choice(available))
        else:
            # Select any other folder uniformly
            available = graph_folders[:-2] if len(graph_folders) > 2 else graph_folders
            if available:
                selected.append(random.choice(available))
            else:
                selected.append(graph_folders[-1])
    return selected

def load_adjacency_matrix_from_file(matrix_filepath):
    """
    Load adjacency matrix from file.
    Expected format: a Python list of lists, e.g., [[0, 1, INF], [1, 0, 2], ...]
    """
    try:
        content = load_file(matrix_filepath)
        if not content:
            return None
        
        # Replace INF with float('inf') for evaluation
        content = content.replace("INF", "float('inf')")
        matrix = ast.literal_eval(content.strip())
        return matrix
    except Exception as e:
        print(f"Error parsing adjacency matrix from {matrix_filepath}: {e}")
        return None

def extract_adjacency_matrix_from_response(response):
    """
    Extract adjacency matrix from LLM response.
    Looks for: ADJACENCY MATRIX: [[...]]
    """
    match = re.search(r"ADJACENCY MATRIX:\s*(\[\[.*?\]\])", response, re.DOTALL)
    if match:
        try:
            matrix_str = match.group(1)
            matrix_str = matrix_str.replace("INF", "float('inf')")
            matrix = ast.literal_eval(matrix_str)
            return matrix
        except Exception as e:
            print(f"Error parsing matrix from response: {e}")
            return None
    return None


def extract_bool_field(response, label):
    match = re.search(rf"{label}:\s*(True|False|true|false)\b", response, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().lower() == "true"


def extract_int_field(response, label):
    match = re.search(rf"{label}:\s*(\d+)\b", response)
    if not match:
        return None
    return int(match.group(1))


def extract_edge_list_from_response(response):
    match = re.search(r"EDGE_LIST:\s*(\[[\s\S]*?\])\s*(?:\n|$)", response)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
        return canonicalize_edge_list(value)
    except Exception as exc:
        print(f"Error parsing edge list from response: {exc}")
        return None


def extract_vertex_degrees_from_response(response):
    match = re.search(r"VERTEX_DEGREES:\s*(\{[\s\S]*?\})\s*(?:\n|$)", response)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
        return canonicalize_vertex_degrees(value)
    except Exception as exc:
        print(f"Error parsing vertex degrees from response: {exc}")
        return None


def extract_algorithm_from_response(response):
    """
    Extract algorithm name from LLM response.
    Looks for: ALGORITHM NAME: <ALGORITHM NAME>
    """
    match = re.search(r"ALGORITHM NAME:\s*(.+?)(?:\n|$)", response)
    if match:
        return match.group(1).strip()
    return None

def extract_explanation_from_response(response):
    """
    Extract explanation from LLM response.
    Looks for: EXPLANATION OF ALGORITHM CHOICE: <EXPLANATION>
    """
    match = re.search(r"EXPLANATION OF ALGORITHM CHOICE:\s*(.+?)(?:\n|$)", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def normalize_string(s):
    """Normalize string for lenient comparison."""
    # Convert to lowercase, remove extra spaces, remove special characters
    return re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()

def canonicalize_algorithm_name(s):
    """Normalize an algorithm name and remove generic suffix words."""
    tokens = normalize_string(s).split()
    generic_tokens = {"algorithm", "algorithms", "method", "methods", "approach", "approaches"}
    tokens = [token for token in tokens if token not in generic_tokens]
    return " ".join(tokens)

def matrix_to_string(matrix):
    """Convert matrix to a readable string with INF tokens."""
    if matrix is None:
        return "None"

    def convert_value(v):
        if isinstance(v, float) and str(v) == 'inf':
            return "INF"
        return str(v)

    rows = []
    for row in matrix:
        rows.append("[" + ", ".join(convert_value(v) for v in row) + "]")
    return "[" + ", ".join(rows) + "]"

def algorithm_matches_reference(algo_name, explanation, problem_category):
    """
    Check if the LLM's algorithm name matches any known algorithm for the
    given problem category, using an LLM to verify robustly.
    """
    if not algo_name:
        return False, None

    valid_algos = GRAPH_ALGORITHMS.get(problem_category, [])
    if not valid_algos:
        return False, None

    prompt = f'''
Determine if the provided algorithm matches any of the valid algorithms for the given problem category.
Problem Category: {problem_category}
Valid Algorithms: {valid_algos}

Proposed Algorithm: {algo_name}
Explanation: {explanation}

If the proposed algorithm is essentially the same as one of the valid algorithms, reply EXACTLY in this format:
YES: <Matched_Valid_Algorithm>
Otherwise, reply exactly:
NO
'''
    import subprocess
    try:
        result = subprocess.run(
            [
                r"C:\Users\ASUS\AppData\Local\agy\bin\agy.exe",
                "--model",
                "gemini-3.7-flash-high",
                "-p",
                prompt,
            ],
            capture_output=True,
            text=True,
        )
        response = result.stdout.strip()
        
        if response.startswith("YES:"):
            matched = response.split("YES:", 1)[1].strip()
            # Ensure we return the exact string from valid_algos if it matches case-insensitively
            for valid in valid_algos:
                if matched.lower() == valid.lower():
                    return True, valid
            return True, matched
        elif response.startswith("YES"):
            return True, algo_name
            
    except Exception as e:
        print(f"Error during LLM verification: {e}")
        
    return False, None

def matrices_equal(matrix1, matrix2):
    """
    Compare two adjacency matrices.
    Handles float('inf') comparison correctly.
    """
    if matrix1 is None or matrix2 is None:
        return False
    
    if len(matrix1) != len(matrix2):
        return False
    
    for i in range(len(matrix1)):
        if len(matrix1[i]) != len(matrix2[i]):
            return False
        for j in range(len(matrix1[i])):
            val1 = matrix1[i][j]
            val2 = matrix2[i][j]
            
            if isinstance(val1, float) and isinstance(val2, float):
                if str(val1) == 'inf' and str(val2) == 'inf':
                    continue
            
            if val1 != val2:
                return False
    
    return True


def graph_metadata_matches(response, graph):
    """Judge all graph metadata in the LLM response against the Graph methods."""
    expected = graph_to_expected_metadata(graph)

    extracted = {
        "is_directed": extract_bool_field(response, "IS_DIRECTED"),
        "number_of_vertices": extract_int_field(response, "NUMBER_OF_VERTICES"),
        "number_of_edges": extract_int_field(response, "NUMBER_OF_EDGES"),
        "has_self_loops": extract_bool_field(response, "HAS_SELF_LOOPS"),
        "has_parallel_edges": extract_bool_field(response, "HAS_PARALLEL_EDGES"),
        "edge_list": extract_edge_list_from_response(response),
        "vertex_degrees": extract_vertex_degrees_from_response(response),
    }

    if any(value is None for value in (
        extracted["is_directed"],
        extracted["number_of_vertices"],
        extracted["number_of_edges"],
        extracted["has_self_loops"],
        extracted["has_parallel_edges"],
        extracted["edge_list"],
        extracted["vertex_degrees"],
    )):
        return False, extracted, expected, "missing metadata field"

    if extracted["is_directed"] != expected["is_directed"]:
        return False, extracted, expected, "is_directed mismatch"
    if extracted["number_of_vertices"] != expected["number_of_vertices"]:
        return False, extracted, expected, "number_of_vertices mismatch"
    if extracted["number_of_edges"] != expected["number_of_edges"]:
        return False, extracted, expected, "number_of_edges mismatch"
    if extracted["has_self_loops"] != expected["has_self_loops"]:
        return False, extracted, expected, "has_self_loops mismatch"
    if extracted["has_parallel_edges"] != expected["has_parallel_edges"]:
        return False, extracted, expected, "has_parallel_edges mismatch"
    if extracted["edge_list"] != expected["edge_list"]:
        return False, extracted, expected, "edge_list mismatch"
    if extracted["vertex_degrees"] != expected["vertex_degrees"]:
        return False, extracted, expected, "vertex_degrees mismatch"

    return True, extracted, expected, "ok"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("GRAPH ALGORITHM TESTER")
    print("="*80)
    
    progress_file_path = PROGRESS_FILE
    state = None
    if os.path.exists(progress_file_path):
        try:
            with open(progress_file_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            print(f"\n[INFO] Loaded progress from {progress_file_path}. Resuming...")
        except Exception as e:
            print(f"  ❌ ERROR loading progress file: {e}. Starting fresh.")
            
    if state:
        combinations = state.get("combinations", [])
        stats = state.get("stats", {
            'total': 0,
            'adjacency_correct': 0,
            'adjacency_wrong': 0,
            'adjacency_error': 0,
            'algorithm_correct': 0,
            'algorithm_wrong': 0,
            'algorithm_error': 0,
        })
        stats.setdefault('failed_adjacency_matrices', [])
        start_idx = state.get("next_idx", 0)
        
        save_to_log("="*80 + "\n")
        save_to_log(f"RESUMING GRAPH ALGORITHM TESTER LOG (from index {start_idx+1})\n")
        save_to_log("="*80 + "\n\n")
    else:
        # Clear log file
        clear_log()
        save_to_log("="*80 + "\n")
        save_to_log("GRAPH ALGORITHM TESTER LOG\n")
        save_to_log("="*80 + "\n\n")
        
        # ========================================================================
        # STEP 1: Collect all graphs
        # ========================================================================
        
        print("\n[STEP 1] Collecting graphs...")
        
        # Directed graphs
        directed_folders = sorted([d for d in os.listdir(DIRECTED_GRAPHS_BASE)
                                   if os.path.isdir(os.path.join(DIRECTED_GRAPHS_BASE, d))])
        
        # Undirected graphs
        undirected_folders = sorted([d for d in os.listdir(UNDIRECTED_GRAPHS_BASE)
                                     if os.path.isdir(os.path.join(UNDIRECTED_GRAPHS_BASE, d))])
        
        print(f"  Found {len(directed_folders)} directed graph folders")
        print(f"  Found {len(undirected_folders)} undirected graph folders")
        
        # Select NUMBER_OF_GRAPHS/2 directed and NUMBER_OF_GRAPHS/2 undirected graphs with bias
        selected_directed = select_graphs_with_bias(directed_folders, NUMBER_OF_GRAPHS//2, LAST_GRAPH_BIAS)
        selected_undirected = select_graphs_with_bias(undirected_folders, NUMBER_OF_GRAPHS//2, LAST_GRAPH_BIAS)
        
        selected_graphs = [
            ('directed', d) for d in selected_directed
        ] + [
            ('undirected', u) for u in selected_undirected
        ]
        
        print(f"\n  Selected graphs (with bias={LAST_GRAPH_BIAS}):")
        for graph_type, folder in selected_graphs:
            print(f"    - {graph_type}/{folder}")
        
        # ========================================================================
        # STEP 2: Collect all problems
        # ========================================================================
        
        print("\n[STEP 2] Collecting problems...")
        
        problem_categories = [d for d in os.listdir(PROBLEMS_BASE)
                             if os.path.isdir(os.path.join(PROBLEMS_BASE, d))]
        
        selected_problems = {}  # {category: [problem_filepath, ...]}
        
        for category in problem_categories:
            category_path = os.path.join(PROBLEMS_BASE, category)
            problem_files = sorted([f for f in os.listdir(category_path)
                                   if f.endswith('.txt')])
            
            # Select PROBLEMS_FROM_EACH_CATEGORY random problems from this category
            selected = random.sample(problem_files, min(PROBLEMS_FROM_EACH_CATEGORY, len(problem_files)))
            selected_problems[category] = [
                os.path.join(category_path, f) for f in selected
            ]
            
            print(f"  {category}: selected {len(selected)} of {len(problem_files)} problems")
        
        total_problems = sum(len(v) for v in selected_problems.values())
        print(f"\n  Total problems selected: {total_problems}")
        
        # ========================================================================
        # STEP 3: Generate all problem-graph combinations
        # ========================================================================
        
        print("\n[STEP 3] Generating problem-graph combinations...")
        
        combinations = []
        for graph_type, graph_folder in selected_graphs:
            base_dir = DIRECTED_GRAPHS_BASE if graph_type == 'directed' else UNDIRECTED_GRAPHS_BASE
            
            for category, problem_paths in selected_problems.items():
                for problem_path in problem_paths:
                    combinations.append({
                        'graph_type': graph_type,
                        'graph_folder': graph_folder,
                        'base_dir': base_dir,
                        'problem_category': category,
                        'problem_path': problem_path,
                    })
        
        total_combinations = len(combinations)
        print(f"  Total combinations: {total_combinations}")
        print(f"  MAX_PROMPTS_COUNT: {MAX_PROMPTS_COUNT if MAX_PROMPTS_COUNT else 'None (all)'}")
        
    
        if MAX_PROMPTS_COUNT:
    
            combinations = random.sample(
                combinations,
                min(MAX_PROMPTS_COUNT, len(combinations))
            )
            print(f"  Limited to: {len(combinations)} combinations")
            
        # Statistics
        stats = {
            'total': 0,
            'adjacency_correct': 0,
            'adjacency_wrong': 0,
            'adjacency_error': 0,
            'metadata_correct': 0,
            'metadata_wrong': 0,
            'metadata_error': 0,
            'algorithm_correct': 0,
            'algorithm_wrong': 0,
            'algorithm_error': 0,
            'failed_adjacency_matrices': [],
        }
        start_idx = 0
        
    # ========================================================================
    # STEP 4: Process combinations
    # ========================================================================
    
    print("\n[STEP 4] Processing combinations...\n")
    
    for idx, combo in enumerate(combinations[start_idx:], start_idx + 1):
        graph_type = combo['graph_type']
        graph_folder = combo['graph_folder']
        base_dir = combo['base_dir']
        problem_category = combo['problem_category']
        problem_path = combo['problem_path']
        
        stats['total'] += 1
        
        # Construct file paths
        graph_filepath = os.path.join(base_dir, graph_folder, f"graph_{graph_folder}.txt")
        matrix_filepath = os.path.join(base_dir, graph_folder, f"adjacency_matrix_{graph_folder}.txt")
        
        problem_name = os.path.basename(problem_path)
        
        print(f"[{idx}/{len(combinations)}] {graph_type}/{graph_folder} + {problem_category}/{problem_name}")
        
        # Load graph and problem using the Graph class as the source of truth.
        graph = Graph.from_file(graph_filepath, directed=(graph_type == "directed"))
        graph_description = graph_to_description(graph)
        problem_description = load_file(problem_path)
        expected_matrix = graph.get_adjacency_matrix()
        
        if not graph_description or not problem_description:
            print(f"  ❌ ERROR: Failed to load graph or problem")
            save_to_log(f"\n[{idx}] {graph_type}/{graph_folder} + {problem_category}/{problem_name}")
            save_to_log(f"  ❌ ERROR: Failed to load files\n")
            continue
        
        # Build prompt
        prompt = GRAPH_PROMPT.format(
            graph_description=graph_description,
            problem_description=problem_description,
        )

        print(prompt)
        
        # Call model (SLM or agy CLI depending on USE_SLM)
        response = call_model(prompt, max_tokens=2048)
        # =====================================================================

        print("LLM's Complete, Unedited Response:", response)
        print("-" * 80)
        
        # PLACEHOLDER: Simulate response (remove when using real LLM)
        # response = "ADJACENCY MATRIX: [[0, 1], [1, 0]]\nALGORITHM NAME: DFS\nEXPLANATION OF ALGORITHM CHOICE: DFS is good."
        
        # Extract components from response
        extracted_matrix = extract_adjacency_matrix_from_response(response)
        extracted_algo = extract_algorithm_from_response(response)
        extracted_explanation = extract_explanation_from_response(response)
        
        # Check adjacency matrix
        if expected_matrix is None:
            print(f"  ❌ ERROR: Expected matrix file not found")
            save_to_log(f"\n[{idx}] {graph_type}/{graph_folder} + {problem_category}/{problem_name}")
            save_to_log(f"  ADJACENCY MATRIX: ❌ ERROR (expected file missing)\n")
            stats['adjacency_error'] += 1
        elif extracted_matrix is None:
            print(f"  ❌ WRONG ADJACENCY MATRIX (extraction failed)")
            print(f"     LLM adjacency matrix: {extracted_matrix}")
            print(f"     Expected adjacency matrix: {matrix_to_string(expected_matrix)}")
            save_to_log(f"\n[{idx}] {graph_type}/{graph_folder} + {problem_category}/{problem_name}")
            save_to_log(f"  ADJACENCY MATRIX: ❌ WRONG (extraction failed)\n")
            save_to_log(f"  LLM adjacency matrix: {extracted_matrix}\n")
            save_to_log(f"  Expected adjacency matrix: {matrix_to_string(expected_matrix)}\n")
            stats['adjacency_wrong'] += 1
            stats['failed_adjacency_matrices'].append(f"{graph_type}/{graph_folder} + {problem_category}/{problem_name}")
        elif matrices_equal(extracted_matrix, expected_matrix):
            print(f"  ✅ CORRECT ADJACENCY MATRIX")
            save_to_log(f"\n[{idx}] {graph_type}/{graph_folder} + {problem_category}/{problem_name}")
            save_to_log(f"  ADJACENCY MATRIX: ✅ CORRECT\n")
            save_to_log(f"  LLM adjacency matrix: {matrix_to_string(extracted_matrix)}\n")
            stats['adjacency_correct'] += 1
        else:
            print(f"  ❌ WRONG ADJACENCY MATRIX")
            print(f"     LLM adjacency matrix: {matrix_to_string(extracted_matrix)}")
            print(f"     Expected adjacency matrix: {matrix_to_string(expected_matrix)}")
            save_to_log(f"\n[{idx}] {graph_type}/{graph_folder} + {problem_category}/{problem_name}")
            save_to_log(f"  ADJACENCY MATRIX: ❌ WRONG\n")
            save_to_log(f"  LLM adjacency matrix: {matrix_to_string(extracted_matrix)}\n")
            save_to_log(f"  Expected adjacency matrix: {matrix_to_string(expected_matrix)}\n")
            stats['adjacency_wrong'] += 1

        # Check full graph metadata
        metadata_ok, extracted_metadata, expected_metadata, reason = graph_metadata_matches(response, graph)
        if metadata_ok:
            print(f"  ✅ CORRECT GRAPH METADATA")
            save_to_log(f"  GRAPH METADATA: ✅ CORRECT\n")
            save_to_log(f"  LLM metadata: {extracted_metadata}\n")
            stats['metadata_correct'] += 1
        else:
            print(f"  ❌ WRONG GRAPH METADATA ({reason})")
            save_to_log(f"  GRAPH METADATA: ❌ WRONG ({reason})\n")
            save_to_log(f"  LLM metadata: {extracted_metadata}\n")
            save_to_log(f"  Expected metadata: {expected_metadata}\n")
            stats['metadata_wrong'] += 1
        
        # Check algorithm
        valid_algos_for_category = GRAPH_ALGORITHMS.get(problem_category, [])

        if extracted_algo is None:
            print(f"  ❌ WRONG ALGORITHM CHOICE (extraction failed)")
            print(f"     LLM algorithm choice: {extracted_algo}")
            print(f"     Valid algorithms for {problem_category}: {valid_algos_for_category}")
            save_to_log(f"  ALGORITHM CHOICE: ❌ WRONG (extraction failed)\n")
            save_to_log(f"  LLM algorithm choice: {extracted_algo}\n")
            save_to_log(f"  Valid algorithms for {problem_category}: {valid_algos_for_category}\n")
            stats['algorithm_wrong'] += 1
        else:
            match, matched_name = algorithm_matches_reference(extracted_algo, extracted_explanation, problem_category)
            if match:
                print(f"  ✅ CORRECT ALGORITHM CHOICE ({matched_name})")
                save_to_log(f"  ALGORITHM CHOICE: ✅ CORRECT ({matched_name})\n")
                save_to_log(f"  LLM algorithm choice: {extracted_algo}\n")
                save_to_log(f"  Valid algorithms for {problem_category}: {valid_algos_for_category}\n")
                stats['algorithm_correct'] += 1
            else:
                print(f"  ❌ WRONG ALGORITHM CHOICE ({extracted_algo})")
                print(f"     LLM algorithm choice: {extracted_algo}")
                print(f"     Valid algorithms for {problem_category}: {valid_algos_for_category}")
                save_to_log(f"  ALGORITHM CHOICE: ❌ WRONG ({extracted_algo})\n")
                save_to_log(f"  LLM algorithm choice: {extracted_algo}\n")
                save_to_log(f"  Valid algorithms for {problem_category}: {valid_algos_for_category}\n")
                stats['algorithm_wrong'] += 1
        
        # Print real-time accuracy
        print(f"    Adj Matrix Accuracy: {stats['adjacency_correct']}/{stats['total']} ({100*stats['adjacency_correct']//stats['total']}%)")
        print(f"    Metadata Accuracy:   {stats['metadata_correct']}/{stats['total']} ({100*stats['metadata_correct']//stats['total']}%)")
        print(f"    Algorithm Accuracy:  {stats['algorithm_correct']}/{stats['total']} ({100*stats['algorithm_correct']//stats['total']}%)")
        print()
        
        # Checkpoint progress
        state_to_save = {
            "combinations": combinations,
            "stats": stats,
            "next_idx": idx
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=4)
    
    # ========================================================================
    # STEP 5: Final statistics
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    final_stats_text = f"""
FINAL STATISTICS
================

Total Combinations Processed: {stats['total']}

ADJACENCY MATRIX:
  ✅ Correct: {stats['adjacency_correct']} ({100*stats['adjacency_correct']//stats['total'] if stats['total'] > 0 else 0}%)
  ❌ Wrong: {stats['adjacency_wrong']}
  ⚠️  Error: {stats['adjacency_error']}

GRAPH METADATA:
  ✅ Correct: {stats['metadata_correct']} ({100*stats['metadata_correct']//stats['total'] if stats['total'] > 0 else 0}%)
  ❌ Wrong: {stats['metadata_wrong']}
  ⚠️  Error: {stats['metadata_error']}

ALGORITHM CHOICE:
  ✅ Correct: {stats['algorithm_correct']} ({100*stats['algorithm_correct']//stats['total'] if stats['total'] > 0 else 0}%)
  ❌ Wrong: {stats['algorithm_wrong']}
  ⚠️  Error: {stats['algorithm_error']}
"""
    
    if stats.get('failed_adjacency_matrices'):
        final_stats_text += "\nFAILED ADJACENCY MATRICES (Returned None):\n"
        for failed in stats['failed_adjacency_matrices']:
            final_stats_text += f"  - {failed}\n"
            
    print(final_stats_text)
    save_to_log(final_stats_text)
    print(f"\nLog file: {LLM_LOG_FILE}")
    save_to_log(f"\n{'='*80}\n")
    
    # Remove progress file upon successful completion
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

if __name__ == "__main__":
    main()