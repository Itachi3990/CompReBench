import os
import random
import re
import ast
import json
from pathlib import Path
import sys
from collections import defaultdict
import csv
import time
from tqdm import tqdm

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
LM_STUDIO_MODEL = "qwen3-4b-thinking-2507"
SLM_TEMPERATURE = 0.3

# Set to None to process all combinations, or an integer to limit prompts
MAX_PROMPTS_COUNT = None  # if None then, ALL prompts are fed to the LLM/SLM; otherwise if some integer value then, only the first MAX_PROMPTS_COUNT prompts are fed to the LLM/SLM, and then the script stops running after that
NUMBER_OF_GRAPHS = None # if None then, ALL graphs are selected; otherwise if integer value then, that many graphs are selected.
NUMBER_OF_PROMPTS_PER_GRAPH = 2 # the LLM or SLM will be tested with the exact same graph this many times (API calls are memoryless anyway)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRECTED_GRAPHS_BASE = os.path.join(BASE_DIR, "graph_samples", "directed")
UNDIRECTED_GRAPHS_BASE = os.path.join(BASE_DIR, "graph_samples", "undirected")
OUTPUT_DIR = os.path.join(BASE_DIR, LM_STUDIO_MODEL)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LLM_LOG_FILE = os.path.join(OUTPUT_DIR, "llm_log_structure.txt")
LLM_CSV_LOG_FILE = os.path.join(OUTPUT_DIR, "llm_log_structure.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_structure.json")

USE_TWINPORT = True # use twinport encoding method (or incident encoding method)

# ============================================================================
# LLM / SLM INFERENCE HELPER
# ============================================================================

def call_model(prompt: str) -> str:
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
                "seed": random.randint(1, 10000000),
                },
        )
        if response.status_code != 200:
            print(f"LM Studio API Error: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        import subprocess
        result = subprocess.run(
            [
                "agy --model",
                "gemini-3.8-flash-high",
                "-p",
                prompt,
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

# Prompt template (modular, easily replaceable)
TWINPORT_EXPLANATION = """Understanding Twin-Port Encoding:
1. The graph consists of Vertices (Nodes) and Edges. Node names are plain strings (e.g., A, B, P, Q, V1, V2, V50 etc).
2. EDGE TICKETS explicitly define each edge with local port addresses:
   - For DIRECTED graphs: "E01: A:p01 -> B:p01 (wt: 7)" means directed edge E01 goes from node A (port p01) to node B (port p01) with weight 7.
   - For UNDIRECTED graphs: "E01: A:p01 <-> B:p01 (wt: 7)" means an undirected edge E01 connects node A (port p01) and node B (port p01) with weight 7.
   - "p01", "p02", etc. refer to local port addresses on the vertices (which act as explicit memory pointers). You only need to extract the actual node names (e.g. A, B, V1, V2, V45, V50 etc) to build the graph structure.
3. INCIDENCE LEDGER shows the same graph from a node-centric view:
   - For DIRECTED graphs: Under "NODE A:", "OUT [p01 -> B (E01)]" indicates outgoing edges from A, and "IN [p03 <- D (E04)]" indicates incoming edges to A.
   - For UNDIRECTED graphs: Under "NODE A:", "[p01 <-> B (E01)]" indicates undirected edges incident to A.
"""

STRUCTURE_PROMPT = """DO NOT USE THE INTERNET.
{encoding_explanation}
Your task is to extract the exact graph metadata based on the provided graph description. You need to determine whether the graph is directed or undirected, whether the graph has any self-loops, and the number of vertices and edges. You also need to determine the adjacency matrix and the edge list of the graph, and the in-degree and out-degree of every vertex. 

Use deterministic formatting (as shown below) and do not add extra labels or commentary.
An example of the exact output format (DO NOT COPY THESE VALUES, it is only an example output format) for a hypothetical random graph with 3 edges and 3 vertices is given below:

IS_DIRECTED: True
NUMBER_OF_VERTICES: 3
NUMBER_OF_EDGES: 3
HAS_SELF_LOOPS: False
EDGE_LIST: [["A", "B", 1], ["B", "C", 3], ["C", "A", 2]]
VERTEX_DEGREES: {{"A": {{"in_degree": 1, "out_degree": 1}}, "B": {{"in_degree": 1, "out_degree": 1}}, "C": {{"in_degree": 1, "out_degree": 1}}}}
ADJACENCY MATRIX: [[0, 1, 0], [0, 0, 3], [2, 0, 0]]

Important instructions:
- Use Python literal formatting for all structured values.
- Boolean values must be written as True or False.
- EDGE_LIST must be a list of [source, destination, weight] triples.
- VERTEX_DEGREES must map each vertex to a dictionary with in_degree and out_degree keys.
- ADJACENCY MATRIX MUST be an N x N matrix (2D list) where N is NUMBER_OF_VERTICES. It MUST have exactly N rows and N columns (e.g. 4x4 if N=4, 5x5 if N=5). DO NOT default to 3x3 unless N=3.
- In ADJACENCY MATRIX, row i and column j correspond to the i-th and j-th vertices in sorted order. If there is an edge from vertex i to vertex j, entry [i][j] is the edge weight, otherwise 0.
- For undirected graphs, in_degree and out_degree should be equal to the undirected degree.
- Sort the EDGE_LIST and VERTEX_DEGREES entries in a deterministic vertex order before writing them.

Now, consider the actual graph described below. Provide ONLY the extracted metadata for THIS graph below, using the exact output format shown above:

{graph_description}
"""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _natural_vertex_key(vertex):
    """
    Natural sort key for vertex labels.
    """
    parts = re.split(r'(\d+)', vertex)
    return tuple(
        (0, int(part)) if part.isdigit()
        else (1, part.lower())
        for part in parts
        if part
    )

def canonicalize_edge_list(edge_list):
    """Normalize edge list output into a deterministic sorted list of triples."""
    normalized = []
    for u, v, weight in edge_list:
        try:
            f_val = float(weight)
            w = int(f_val) if f_val.is_integer() else f_val
        except (ValueError, TypeError):
            w = 1
        normalized.append((str(u), str(v), w))
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

def graph_to_expected_metadata(graph: Graph) -> dict:
    """Return canonical metadata from the Graph object that the LLM is expected to emit."""
    if not isinstance(graph, Graph):
        raise TypeError("graph_to_expected_metadata expects a Graph instance.")

    return {
        "is_directed": bool(graph.is_directed()),
        "number_of_vertices": int(graph.get_number_of_vertices()),
        "number_of_edges": int(graph.get_number_of_edges()),
        "has_self_loops": bool(graph.has_self_loops()),
        "edge_list": canonicalize_edge_list(graph.get_edge_list()),
        "vertex_degrees": canonicalize_vertex_degrees(graph.get_indegree_and_outdegree_of_every_vertex()),
    }

def log_to_csv(filepath, row_dict, fieldnames):
    file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
    try:
        with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def save_to_log(message):
    """Append message to LLM log file."""
    with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def clear_log():
    """Clear the LLM log file at start."""
    with open(LLM_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")
    with open(LLM_CSV_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

def select_graphs(graph_folders, count):
    """
    If 'count' is None, return all graph folders.
    Else, select 'count' graph folders uniformly at random.
    """
    if not graph_folders:
        return []
    if count is None:
        return list(graph_folders)
    return [random.choice(graph_folders) for _ in range(count)]


def extract_adjacency_matrix_from_response(response):
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



def graph_metadata_matches(response, graph):
    """Judge all graph metadata in the LLM response against the Graph methods."""
    expected = graph_to_expected_metadata(graph)

    extracted = {
        "is_directed": extract_bool_field(response, "IS_DIRECTED"),
        "number_of_vertices": extract_int_field(response, "NUMBER_OF_VERTICES"),
        "number_of_edges": extract_int_field(response, "NUMBER_OF_EDGES"),
        "has_self_loops": extract_bool_field(response, "HAS_SELF_LOOPS"),
        "edge_list": extract_edge_list_from_response(response),
        "vertex_degrees": extract_vertex_degrees_from_response(response),
    }

    if any(value is None for value in (
        extracted["is_directed"],
        extracted["number_of_vertices"],
        extracted["number_of_edges"],
        extracted["has_self_loops"],
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
    if extracted["edge_list"] != expected["edge_list"]:
        return False, extracted, expected, "edge_list mismatch"
    if extracted["vertex_degrees"] != expected["vertex_degrees"]:
        return False, extracted, expected, "vertex_degrees mismatch"

    return True, extracted, expected, "ok"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_slm_inference():
    print("="*80)
    print("GRAPH STRUCTURE TESTER - SLM INFERENCE")
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
            
    unjudged_graphs = state.get("unjudged_graphs", []) if state else []
    if state:
        combinations = state.get("combinations", [])
        start_idx = state.get("next_idx", 0)
        
        save_to_log("="*80 + "\n")
        save_to_log(f"RESUMING GRAPH STRUCTURE TESTER LOG (from index {start_idx+1})\n")
        save_to_log("="*80 + "\n\n")
    else:
        clear_log()
        save_to_log("="*80 + "\n")
        save_to_log("GRAPH STRUCTURE TESTER LOG\n")
        save_to_log("="*80 + "\n\n")
        
        print("\n[STEP 1] Collecting graphs...")
        
        directed_files = sorted([f for f in os.listdir(DIRECTED_GRAPHS_BASE)
                                 if f.endswith(".txt") and os.path.isfile(os.path.join(DIRECTED_GRAPHS_BASE, f))])
        
        undirected_files = sorted([f for f in os.listdir(UNDIRECTED_GRAPHS_BASE)
                                   if f.endswith(".txt") and os.path.isfile(os.path.join(UNDIRECTED_GRAPHS_BASE, f))])
        
        print(f"  Found {len(directed_files)} directed graphs")
        print(f"  Found {len(undirected_files)} undirected graphs")
        
        directed_count = None if NUMBER_OF_GRAPHS is None else NUMBER_OF_GRAPHS // 2
        undirected_count = None if NUMBER_OF_GRAPHS is None else NUMBER_OF_GRAPHS // 2
        
        selected_directed = select_graphs(directed_files, directed_count)
        selected_undirected = select_graphs(undirected_files, undirected_count)
        
        selected_graphs = [
            ('directed', d) for d in selected_directed
        ] + [
            ('undirected', u) for u in selected_undirected
        ]
        
        print(f"\n  Selected graphs:")
        for graph_type, filename in selected_graphs:
            print(f"    - {graph_type}/{filename}")
            
        combinations = []
        for attempt in range(1, NUMBER_OF_PROMPTS_PER_GRAPH + 1):
            for graph_type, graph_file in selected_graphs:
                base_dir = DIRECTED_GRAPHS_BASE if graph_type == 'directed' else UNDIRECTED_GRAPHS_BASE
                combinations.append({
                    'graph_type': graph_type,
                    'graph_file': graph_file,
                    'base_dir': base_dir,
                    'attempt': attempt,
                })
            
        total_combinations = len(combinations)
        print(f"\n  Total prompts to be sent: {total_combinations}")
        print(f"  MAX_PROMPTS_COUNT: {MAX_PROMPTS_COUNT if MAX_PROMPTS_COUNT else 'None (all)'}")
        
        if MAX_PROMPTS_COUNT:
            combinations = combinations[:MAX_PROMPTS_COUNT]
            print(f"  Limited to: {len(combinations)} prompts")
            
        start_idx = 0
        
    print("\n[STEP 2] Processing graphs...\n")
    
    progress_bar = tqdm(
        enumerate(combinations[start_idx:], start_idx + 1),
        total=len(combinations),
        initial=start_idx,
        desc="Processing graphs",
        dynamic_ncols=True,
    )
    for idx, combo in progress_bar:
        graph_type = combo['graph_type']
        graph_file = combo.get('graph_file') or combo.get('graph_folder')
        base_dir = combo['base_dir']
        
        if graph_file.endswith(".txt"):
            graph_filepath = os.path.join(base_dir, graph_file)
            graph_name = os.path.splitext(graph_file)[0]
        else:
            graph_name = f"graph_{str(graph_file).zfill(2)}"
            graph_filepath = os.path.join(base_dir, f"{graph_name}.txt")
            if not os.path.exists(graph_filepath):
                graph_filepath = os.path.join(base_dir, str(graph_file), f"graph_{str(graph_file).zfill(2)}.txt")
        
        print(f"[{idx}/{len(combinations)}] {graph_type}/{graph_name}")
        
        graph = Graph.from_file(graph_filepath, directed=(graph_type == "directed"))
        
        # Determine encoding method and build prompt
        encoding_method = "twinport" if USE_TWINPORT else "incident"
        fallback_used = False
        inference_success = False
        error_msg = ""
        
        if USE_TWINPORT:
            graph_description = graph.twinport_description()
            explanation = "\n" + TWINPORT_EXPLANATION
        else:
            graph_description = graph.incident_description()
            explanation = ""
        
        if not graph_description:
            print(f"  ❌ ERROR: Failed to load graph")
            save_to_log(f"\n[{idx}] {graph_type}/{graph_name}")
            save_to_log(f"  ❌ ERROR: Failed to load files\n")
            continue
        
        prompt = STRUCTURE_PROMPT.format(
            encoding_explanation=explanation,
            graph_description=graph_description,
        )

        print(prompt) # print prompt to console
        
        attempt = combo.get('attempt', 1)
        start_time = time.time()
        try:
            response = call_model(prompt)
            inference_success = True
            print("LLM's Complete, Unedited Response:", response)
            print("-" * 80)
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ {encoding_method.capitalize()} inference failed: {e}")

            # If twinport failed, fall back to incident encoding even if USE_TWINPORT is True
            if USE_TWINPORT:
                print(f"  🔄 Retrying with INCIDENT encoding method for {graph_type}/{graph_name}...")
                fallback_desc = graph.incident_description()
                fallback_prompt = STRUCTURE_PROMPT.format(
                    encoding_explanation="",
                    graph_description=fallback_desc,
                )
                try:
                    response = call_model(fallback_prompt)
                    inference_success = True
                    fallback_used = True
                    encoding_method = "incident (fallback)"
                    prompt = fallback_prompt
                    print("  ✅ Incident encoding fallback succeeded! LLM Response:", response)
                    print("-" * 80)
                except Exception as fallback_e:
                    error_msg = f"Twinport failed ({e}) | Incident fallback failed ({fallback_e})"
                    response = f"ERROR: {error_msg}"
                    print(f"  ❌ Incident encoding fallback ALSO failed: {fallback_e}")
                    print("-" * 80)
            else:
                response = f"ERROR: {error_msg}"
                print("-" * 80)

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        if not inference_success:
            unjudged_graphs.append({
                "index": idx,
                "graph_type": graph_type,
                "graph_name": graph_name,
                "attempt": attempt,
                "nodes": len(graph.vertices),
                "edges": len(graph.edges),
                "reason": error_msg,
            })

        save_to_log(f"\n[{idx} - Attempt {attempt}] {graph_type}/{graph_name} (Encoding: {encoding_method})")
        save_to_log(f"Prompt to the LLM: \n{prompt}")
        save_to_log(f"LLM's Complete, Unedited Response: \n{response}")

        csv_row = {
            "Attempt": attempt,
            "Graph Type": graph_type,
            "Graph Folder": graph_name,
            "Execution Time (s)": duration,
            "Number of Nodes": len(graph.vertices),
            "Number of Edges": len(graph.edges),
            "Adjacency Match": "",
            "Adjacency Error": "",
            "Adjacency Entry Match Score": "",
            "Adjacency Row Match Score": "",
            "LLM Adjacency Matrix": "",
            "Expected Adjacency Matrix": "",
            "Metadata Match": "",
            "LLM Metadata": "",
            "Expected Metadata": "",
            "Metadata Error Reason": "",
            "Prompt Length (chars)": len(prompt),
            "Response Length (chars)": len(response),
            "Raw Response": response
        }
        fieldnames = list(csv_row.keys())
        log_to_csv(LLM_CSV_LOG_FILE, csv_row, fieldnames)

        state_to_save = {
            "combinations": combinations,
            "next_idx": idx,
            "unjudged_graphs": unjudged_graphs,
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=4)
    
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    print("\n✅ SLM Inference completed.")

    # ── Report on unjudged / context-exceeded graphs ──────────────────────────
    if unjudged_graphs:
        unjudged_report = format_unjudged_graphs_report(unjudged_graphs)
        print(unjudged_report)
        save_to_log(unjudged_report)


# ============================================================================
# SCORING SCHEME  (100-point rubric)
# ============================================================================
#  10 pts  → adjacency matrix ENTRY score  (get_adjacency_matrix_entry_score)
#  20 pts  → adjacency matrix ROW   score  (get_adjacency_matrix_row_score)
#  30 pts  → edge list score               (get_edge_list_score)   — partial credit
#  30 pts  → vertex degrees score          (get_vertex_degrees_score) — partial credit
#  10 pts  → other metadata               (is_directed, n_vertices, n_edges, has_self_loops)

def _score_other_metadata(response, graph):
    """Return (score_0_to_1, detail_dict) for the 4 simple metadata fields."""
    fields = {
        "is_directed":        (extract_bool_field(response, "IS_DIRECTED"),        bool(graph.is_directed())),
        "number_of_vertices": (extract_int_field(response, "NUMBER_OF_VERTICES"),  int(graph.get_number_of_vertices())),
        "number_of_edges":    (extract_int_field(response, "NUMBER_OF_EDGES"),      int(graph.get_number_of_edges())),
        "has_self_loops":     (extract_bool_field(response, "HAS_SELF_LOOPS"),      bool(graph.has_self_loops())),
    }
    correct = sum(1 for llm_val, exp_val in fields.values() if llm_val == exp_val)
    score = correct / len(fields)
    detail = {k: {"llm": v[0], "expected": v[1], "ok": v[0] == v[1]} for k, v in fields.items()}
    return score, detail


def score_response(response: str, graph) -> dict:
    """
    Score a single LLM response against the ground-truth graph.
    Returns a dict with all sub-scores and a final 0–100 total.
    """
    # --- Adjacency matrix ---
    extracted_matrix = extract_adjacency_matrix_from_response(response)
    expected_matrix  = graph.get_adjacency_matrix()
    adj_entry_raw = graph.get_adjacency_matrix_entry_score(extracted_matrix) if extracted_matrix is not None else 0.0
    adj_row_raw   = graph.get_adjacency_matrix_row_score(extracted_matrix)   if extracted_matrix is not None else 0.0
    adj_match     = graph.adjacency_matrix_matches(extracted_matrix) if extracted_matrix is not None else False

    # --- Edge list ---
    extracted_edges = extract_edge_list_from_response(response)
    edge_score_raw  = graph.get_edge_list_score(extracted_edges) if extracted_edges is not None else 0.0
    expected_edges  = canonicalize_edge_list(graph.get_edge_list())

    # --- Vertex degrees ---
    extracted_degrees = extract_vertex_degrees_from_response(response)
    deg_score_raw     = graph.get_vertex_degrees_score(extracted_degrees) if extracted_degrees is not None else 0.0
    expected_degrees  = canonicalize_vertex_degrees(graph.get_indegree_and_outdegree_of_every_vertex())

    # --- Other metadata ---
    meta_score_raw, meta_detail = _score_other_metadata(response, graph)

    # --- Weighted total (out of 100) ---
    total = (
        adj_entry_raw * 10
        + adj_row_raw   * 20
        + edge_score_raw * 30
        + deg_score_raw  * 30
        + meta_score_raw * 10
    )

    return {
        "adj_entry_score":  round(adj_entry_raw * 10, 2),
        "adj_row_score":    round(adj_row_raw   * 20, 2),
        "edge_list_score":  round(edge_score_raw * 30, 2),
        "deg_score":        round(deg_score_raw  * 30, 2),
        "meta_score":       round(meta_score_raw * 10, 2),
        "total_score":      round(total, 2),
        "adj_match":        adj_match,
        "extracted_matrix": extracted_matrix,
        "expected_matrix":  expected_matrix,
        "extracted_edges":  extracted_edges,
        "expected_edges":   expected_edges,
        "extracted_degrees": extracted_degrees,
        "expected_degrees":  expected_degrees,
        "meta_detail":      meta_detail,
    }


def _fmt_matrix(matrix):
    """Pretty-print a matrix for human comparison."""
    if matrix is None:
        return "  (could not parse)"
    return "\n".join("  " + str(row) for row in matrix)


def _fmt_edges(edges):
    if not edges:
        return "  (none / could not parse)"
    return "\n".join(f"  {e}" for e in edges)


def _fmt_degrees(degrees):
    if not degrees:
        return "  (none / could not parse)"
    lines = []
    for v, d in degrees.items():
        lines.append(f"  {v}: in={d.get('in_degree','?')}  out={d.get('out_degree','?')}")
    return "\n".join(lines)


def format_unjudged_graphs_report(unjudged_list) -> str:
    """Format a comprehensive summary report of graphs that could not be evaluated."""
    if not unjudged_list:
        return (
            "\n" + "=" * 92 + "\n"
            + "ALL GRAPHS PROCESSED SUCCESSFULLY — NO UNJUDGED / EXCEEDED GRAPHS\n"
            + "=" * 92 + "\n"
        )

    lines = [
        "",
        "=" * 92,
        "UNJUDGED / CONTEXT EXCEEDED GRAPHS REPORT",
        "=" * 92,
        f"Total graphs/attempts that could not be evaluated: {len(unjudged_list)}",
        "-" * 92,
        f"{'Idx':<6} {'Type':<12} {'Graph Name':<16} {'Attempt':<9} {'Nodes':<8} {'Edges':<8} {'Reason':<30}",
        "-" * 92,
    ]
    for item in unjudged_list:
        reason_short = str(item.get('reason', ''))
        if len(reason_short) > 35:
            reason_short = reason_short[:32] + "..."
        lines.append(
            f"{str(item.get('index', '?')):<6} {str(item.get('graph_type', '')):<12} {str(item.get('graph_name', '')):<16} "
            f"{str(item.get('attempt', '')):<9} {str(item.get('nodes', '')):<8} {str(item.get('edges', '')):<8} {reason_short:<30}"
        )
    lines.append("=" * 92)
    lines.append("")
    return "\n".join(lines)


def generate_comparative_analysis(df) -> str:
    """
    Generate comparative analysis for:
      - Small graphs (< 20 nodes)
      - Medium graphs (20-50 nodes)
      - Large graphs (> 50 nodes)
    """
    import ast
    import pandas as pd

    def _extract_nodes(row):
        val = row.get("Number of Nodes")
        if pd.notna(val) and str(val).strip() != "":
            try:
                return int(float(val))
            except (ValueError, TypeError):
                pass
        deg_str = row.get("Expected Vertex Degrees")
        if pd.notna(deg_str) and str(deg_str).strip() != "":
            try:
                parsed = ast.literal_eval(str(deg_str))
                if isinstance(parsed, dict):
                    return len(parsed)
            except Exception:
                pass
        return None

    # Filter to judged rows with numeric Total Score
    valid_rows = []
    for _, row in df.iterrows():
        total_sc = row.get("Total Score (100)")
        if pd.notna(total_sc) and str(total_sc).strip() != "":
            try:
                sc_float = float(total_sc)
                nodes = _extract_nodes(row)
                if nodes is not None:
                    valid_rows.append({
                        "nodes": nodes,
                        "total_score": sc_float,
                        "adj_entry": float(row.get("Adj Entry Score (10)", 0) or 0),
                        "adj_row": float(row.get("Adj Row Score (20)", 0) or 0),
                        "edge_score": float(row.get("Edge List Score (30)", 0) or 0),
                        "deg_score": float(row.get("Degree Score (30)", 0) or 0),
                        "meta_score": float(row.get("Other Meta Score (10)", 0) or 0),
                        "adj_match": str(row.get("Adj Exact Match", "")).strip().lower() == "true",
                    })
            except (ValueError, TypeError):
                continue

    tiers = {
        "Small (<20)": [r for r in valid_rows if r["nodes"] < 20],
        "Medium (20-50)": [r for r in valid_rows if 20 <= r["nodes"] <= 50],
        "Large (>50)": [r for r in valid_rows if r["nodes"] > 50],
    }

    def _stats(group):
        count = len(group)
        if count == 0:
            return {
                "count": 0,
                "avg_total": 0.0,
                "perfect_cnt": 0,
                "perfect_pct": 0.0,
                "above80_cnt": 0,
                "above80_pct": 0.0,
                "adj_match_cnt": 0,
                "adj_match_pct": 0.0,
                "avg_adj_entry": 0.0,
                "avg_adj_row": 0.0,
                "avg_edge": 0.0,
                "avg_deg": 0.0,
                "avg_meta": 0.0,
            }
        p_cnt = sum(1 for r in group if r["total_score"] >= 100.0)
        a80_cnt = sum(1 for r in group if r["total_score"] >= 80.0)
        m_cnt = sum(1 for r in group if r["adj_match"])
        return {
            "count": count,
            "avg_total": sum(r["total_score"] for r in group) / count,
            "perfect_cnt": p_cnt,
            "perfect_pct": (p_cnt / count) * 100,
            "above80_cnt": a80_cnt,
            "above80_pct": (a80_cnt / count) * 100,
            "adj_match_cnt": m_cnt,
            "adj_match_pct": (m_cnt / count) * 100,
            "avg_adj_entry": sum(r["adj_entry"] for r in group) / count,
            "avg_adj_row": sum(r["adj_row"] for r in group) / count,
            "avg_edge": sum(r["edge_score"] for r in group) / count,
            "avg_deg": sum(r["deg_score"] for r in group) / count,
            "avg_meta": sum(r["meta_score"] for r in group) / count,
        }

    sm = _stats(tiers["Small (<20)"])
    med = _stats(tiers["Medium (20-50)"])
    lg = _stats(tiers["Large (>50)"])

    def _cell_score(st, key, max_val):
        if st["count"] == 0:
            return "N/A (0 graphs)"
        return f"{st[key]:.2f} / {max_val} ({(st[key]/max_val)*100:.1f}%)"

    def _cell_rate(st, cnt_key, pct_key):
        if st["count"] == 0:
            return "N/A (0 graphs)"
        return f"{st[cnt_key]}/{st['count']} ({st[pct_key]:.1f}%)"

    def _cell_total(st):
        if st["count"] == 0:
            return "N/A (0 graphs)"
        return f"{st['avg_total']:.2f} / 100 ({st['avg_total']:.1f}%)"

    lines = [
        "",
        "=" * 92,
        "COMPARATIVE ANALYSIS BY GRAPH SIZE",
        "=" * 92,
        "  • Small Graphs  : < 20 nodes",
        "  • Medium Graphs : 20 - 50 nodes",
        "  • Large Graphs  : > 50 nodes",
        "-" * 92,
        f"{'Metric':<38} {'Small (<20)':<25} {'Medium (20-50)':<25} {'Large (>50)':<25}",
        "-" * 92,
        f"{'Total Evaluated Attempts':<38} {sm['count']:<25} {med['count']:<25} {lg['count']:<25}",
        f"{'Average Accuracy / Total Score':<38} {_cell_total(sm):<25} {_cell_total(med):<25} {_cell_total(lg):<25}",
        f"{'High Accuracy Rate (≥ 80/100)':<38} {_cell_rate(sm, 'above80_cnt', 'above80_pct'):<25} {_cell_rate(med, 'above80_cnt', 'above80_pct'):<25} {_cell_rate(lg, 'above80_cnt', 'above80_pct'):<25}",
        f"{'Perfect Score Rate (100/100)':<38} {_cell_rate(sm, 'perfect_cnt', 'perfect_pct'):<25} {_cell_rate(med, 'perfect_cnt', 'perfect_pct'):<25} {_cell_rate(lg, 'perfect_cnt', 'perfect_pct'):<25}",
        f"{'Adjacency Exact Match Rate':<38} {_cell_rate(sm, 'adj_match_cnt', 'adj_match_pct'):<25} {_cell_rate(med, 'adj_match_cnt', 'adj_match_pct'):<25} {_cell_rate(lg, 'adj_match_cnt', 'adj_match_pct'):<25}",
        "-" * 92,
        "DETAILED SUB-SCORE BREAKDOWN (Average Score & Percentage of Max):",
        f"{'  • Adj Entry Score (max 10)':<38} {_cell_score(sm, 'avg_adj_entry', 10):<25} {_cell_score(med, 'avg_adj_entry', 10):<25} {_cell_score(lg, 'avg_adj_entry', 10):<25}",
        f"{'  • Adj Row Score (max 20)':<38} {_cell_score(sm, 'avg_adj_row', 20):<25} {_cell_score(med, 'avg_adj_row', 20):<25} {_cell_score(lg, 'avg_adj_row', 20):<25}",
        f"{'  • Edge List Score (max 30)':<38} {_cell_score(sm, 'avg_edge', 30):<25} {_cell_score(med, 'avg_edge', 30):<25} {_cell_score(lg, 'avg_edge', 30):<25}",
        f"{'  • Vertex Degree Score (max 30)':<38} {_cell_score(sm, 'avg_deg', 30):<25} {_cell_score(med, 'avg_deg', 30):<25} {_cell_score(lg, 'avg_deg', 30):<25}",
        f"{'  • Other Metadata (max 10)':<38} {_cell_score(sm, 'avg_meta', 10):<25} {_cell_score(med, 'avg_meta', 10):<25} {_cell_score(lg, 'avg_meta', 10):<25}",
        "=" * 92,
    ]
    return "\n".join(lines)


def run_graph_structure_judge():
    print("=" * 80)
    print("GRAPH STRUCTURE JUDGE  (100-point rubric)")
    print("=" * 80)

    import pandas as pd

    if not os.path.exists(LLM_CSV_LOG_FILE):
        print(f"❌ CSV log file not found at {LLM_CSV_LOG_FILE}. Run slm inference first.")
        return

    df = pd.read_csv(LLM_CSV_LOG_FILE)

    # ── Ensure all scoring columns exist ──────────────────────────────────────
    score_cols = [
        "Total Score (100)",
        "Adj Entry Score (10)",
        "Adj Row Score (20)",
        "Edge List Score (30)",
        "Degree Score (30)",
        "Other Meta Score (10)",
        "Adj Exact Match",
        "LLM Adjacency Matrix",
        "Expected Adjacency Matrix",
        "LLM Edge List",
        "Expected Edge List",
        "LLM Vertex Degrees",
        "Expected Vertex Degrees",
        "Meta Detail",
    ]
    for col in score_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    total_rows = len(df)
    score_accumulator = []   # collect total scores for final averages

    judge_log_lines = []
    judge_log_lines.append("=" * 80)
    judge_log_lines.append("GRAPH STRUCTURE JUDGE — DETAILED LOG")
    judge_log_lines.append("=" * 80)

    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Judging graphs", dynamic_ncols=True):
        # Skip already-judged rows
        existing = row.get("Total Score (100)", None)
        if pd.notna(existing) and str(existing).strip() != "":
            try:
                score_accumulator.append(float(existing))
            except ValueError:
                pass
            continue

        graph_type   = row["Graph Type"]
        graph_name   = str(row["Graph Folder"]).strip()
        attempt      = row["Attempt"]
        print(f"Judging row {index + 1}/{total_rows}: {graph_type}/{graph_name} Attempt {attempt}")

        base_dir      = DIRECTED_GRAPHS_BASE if graph_type == "directed" else UNDIRECTED_GRAPHS_BASE

        # Resolve graph filepath: direct file, with .txt, with graph_ prefix, or legacy nested folder
        if os.path.exists(os.path.join(base_dir, graph_name)):
            graph_filepath = os.path.join(base_dir, graph_name)
        elif os.path.exists(os.path.join(base_dir, f"{graph_name}.txt")):
            graph_filepath = os.path.join(base_dir, f"{graph_name}.txt")
        elif os.path.exists(os.path.join(base_dir, f"graph_{graph_name.zfill(2)}.txt")):
            graph_filepath = os.path.join(base_dir, f"graph_{graph_name.zfill(2)}.txt")
        else:
            graph_filepath = os.path.join(base_dir, graph_name.zfill(2), f"graph_{graph_name.zfill(2)}.txt")

        try:
            graph = Graph.from_file(graph_filepath, directed=(graph_type == "directed"))
        except Exception as e:
            print(f"  ❌ ERROR loading graph: {e}")
            continue

        response = str(row["Raw Response"])
        if response.startswith("ERROR:") or response.strip() == "" or response == "None":
            print(f"  ⚠️ Skipping row {index + 1}: Inference had previously failed ({response[:50]}...)")
            df.at[index, "Number of Nodes"] = len(graph.vertices)
            df.at[index, "Number of Edges"] = len(graph.edges)
            df.at[index, "Total Score (100)"] = ""
            df.at[index, "Metadata Error Reason"] = "Inference Error / Context Exceeded"
            continue

        sc = score_response(response, graph)
        score_accumulator.append(sc["total_score"])

        # ── Write to CSV ──────────────────────────────────────────────────────
        df.at[index, "Number of Nodes"]        = len(graph.vertices)
        df.at[index, "Number of Edges"]        = len(graph.edges)
        df.at[index, "Total Score (100)"]      = sc["total_score"]
        df.at[index, "Adj Entry Score (10)"]   = sc["adj_entry_score"]
        df.at[index, "Adj Row Score (20)"]     = sc["adj_row_score"]
        df.at[index, "Edge List Score (30)"]   = sc["edge_list_score"]
        df.at[index, "Degree Score (30)"]      = sc["deg_score"]
        df.at[index, "Other Meta Score (10)"]  = sc["meta_score"]
        df.at[index, "Adj Exact Match"]        = sc["adj_match"]
        df.at[index, "LLM Adjacency Matrix"]   = matrix_to_string(sc["extracted_matrix"])
        df.at[index, "Expected Adjacency Matrix"] = matrix_to_string(sc["expected_matrix"])
        df.at[index, "LLM Edge List"]          = str(sc["extracted_edges"])
        df.at[index, "Expected Edge List"]     = str(sc["expected_edges"])
        df.at[index, "LLM Vertex Degrees"]     = str(sc["extracted_degrees"])
        df.at[index, "Expected Vertex Degrees"] = str(sc["expected_degrees"])
        df.at[index, "Meta Detail"]            = json.dumps(sc["meta_detail"])
        df.to_csv(LLM_CSV_LOG_FILE, index=False)

        # ── Build detailed TXT log entry ──────────────────────────────────────
        sep = "-" * 60
        entry = [
            "",
            "=" * 80,
            f"GRAPH: {graph_type}/{graph_name}   Attempt: {attempt}",
            f"TOTAL SCORE: {sc['total_score']:.2f} / 100",
            "=" * 80,
            "",
            "── SCORE BREAKDOWN ─────────────────────────────────────────",
            f"  Adjacency Entry Score  (10 pts max): {sc['adj_entry_score']:.2f}",
            f"  Adjacency Row Score    (20 pts max): {sc['adj_row_score']:.2f}",
            f"  Edge List Score        (30 pts max): {sc['edge_list_score']:.2f}",
            f"  Vertex Degree Score    (30 pts max): {sc['deg_score']:.2f}",
            f"  Other Metadata Score   (10 pts max): {sc['meta_score']:.2f}",
            "",
            "── ADJACENCY MATRIX ─────────────────────────────────────────",
            f"  Exact match: {sc['adj_match']}",
            "",
            "  LLM output:",
            _fmt_matrix(sc["extracted_matrix"]),
            "",
            "  Expected (ground truth):",
            _fmt_matrix(sc["expected_matrix"]),
            "",
            "── EDGE LIST ────────────────────────────────────────────────",
            "  LLM output:",
            _fmt_edges(sc["extracted_edges"]),
            "",
            "  Expected (ground truth):",
            _fmt_edges(sc["expected_edges"]),
            "",
            "── VERTEX DEGREES ───────────────────────────────────────────",
            "  LLM output:",
            _fmt_degrees(sc["extracted_degrees"]),
            "",
            "  Expected (ground truth):",
            _fmt_degrees(sc["expected_degrees"]),
            "",
            "── OTHER METADATA ───────────────────────────────────────────",
        ]
        for field, info in sc["meta_detail"].items():
            status = "✅" if info["ok"] else "❌"
            entry.append(f"  {status} {field}: LLM={info['llm']}  Expected={info['expected']}")
        entry.append("")

        block = "\n".join(entry)
        judge_log_lines.append(block)
        print(f"  => {sc['total_score']:.2f}/100  (entry={sc['adj_entry_score']}, row={sc['adj_row_score']}, "
              f"edge={sc['edge_list_score']}, deg={sc['deg_score']}, meta={sc['meta_score']})")

    # ── Final statistics ──────────────────────────────────────────────────────
    n = len(score_accumulator)
    avg = sum(score_accumulator) / n if n > 0 else 0.0
    perfect = sum(1 for s in score_accumulator if s >= 100.0)
    above80 = sum(1 for s in score_accumulator if s >= 80.0)

    final_block = "\n".join([
        "",
        "=" * 80,
        "FINAL STATISTICS",
        "=" * 80,
        f"Total rows judged : {n}",
        f"Average score     : {avg:.2f} / 100",
        f"Perfect (100/100) : {perfect}  ({100*perfect//n if n else 0}%)",
        f"≥ 80 / 100        : {above80}  ({100*above80//n if n else 0}%)",
        "",
    ])

    print(final_block)
    judge_log_lines.append(final_block)

    # ── Comparative Analysis by Graph Size ────────────────────────────────────
    comparative_analysis_str = generate_comparative_analysis(df)
    print(comparative_analysis_str)
    judge_log_lines.append(comparative_analysis_str)

    # ── Report of Unjudged / Exceeded Graphs ──────────────────────────────────
    unjudged_in_judge = []
    for index, row in df.iterrows():
        raw = str(row.get("Raw Response", ""))
        tot = row.get("Total Score (100)")
        if raw.startswith("ERROR:") or pd.isna(tot) or str(tot).strip() == "":
            unjudged_in_judge.append({
                "index": index + 1,
                "graph_type": row.get("Graph Type"),
                "graph_name": row.get("Graph Folder"),
                "attempt": row.get("Attempt"),
                "nodes": row.get("Number of Nodes"),
                "edges": row.get("Number of Edges"),
                "reason": raw if raw.startswith("ERROR:") else "Unjudged / missing response",
            })

    if unjudged_in_judge:
        unjudged_report = format_unjudged_graphs_report(unjudged_in_judge)
        print(unjudged_report)
        judge_log_lines.append(unjudged_report)

    full_log = "\n".join(judge_log_lines)
    save_to_log(full_log)
    print(f"\n📄 Detailed log written to: {LLM_LOG_FILE}")


if __name__ == "__main__":
    # Comment out either function as needed
    run_slm_inference()
    run_graph_structure_judge()