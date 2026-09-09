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
SLM_TEMPERATURE = 0.3

# Set to None to process all combinations, or an integer to limit prompts
MAX_PROMPTS_COUNT = 3  # if None then, ALL prompts are generated; otherwise if integer value then, that many prompts are generated, and then the script stops running
NUMBER_OF_GRAPHS = 10 # if None then, ALL graphs are selected; otherwise if integer value then, that many graphs are selected.
NUMBER_OF_PROMPTS_PER_GRAPH = 3 # the LLM or SLM will be tested with the exact same graph this many times (API calls are memoryless anyway)

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
GRAPH_PROMPT = """DO NOT USE THE INTERNET.

Consider the graph described below, which is encoded using the Twin-Port encoding scheme:

Understanding Twin-Port Encoding:
1. The graph consists of Vertices (Nodes) and Edges. Node names are plain strings (e.g., A, B, P, Q).
2. EDGE TICKETS explicitly define each edge:
   - "EDGE: E01" means the edge ID is E01.
   - For DIRECTED graphs: "TAIL_NODE: A" and "HEAD_NODE: B" means the edge goes from vertex A to vertex B.
   - For UNDIRECTED graphs: "NODE1: A" and "NODE2: B" means there is an undirected edge between vertex A and vertex B.
   - "TAIL_PORT: p01" or "PORT1: p02" refer to local port addresses on the vertices (which act as explicit memory pointers). You only need to extract the actual node names (e.g. A, B) to build the graph structure.
   - "WEIGHT: +7.000" specifies the edge weight.
3. INCIDENCE LEDGER shows the same graph from a node-centric view:
   - Under "NODE A", an entry like "OUT p01 -> B [E01]" means vertex A has an outgoing edge E01 to vertex B.

{graph_description}

Extract the exact graph metadata based on the description above. You need to determine whether the graph is directed or undirected, whether the graph has any self-loops, and the number of vertices and edges. You also need to determine the adjacency matrix and the edge list of the graph, and the in-degree and out-degree of every vertex. Use deterministic formatting and do not add extra labels or commentary.

Your output must EXACTLY follow this format, in this exact order:

ADJACENCY MATRIX: [[0, 1, 0], [0, 0, 3], [2, 0, 0]]
IS_DIRECTED: True
NUMBER_OF_VERTICES: 3
NUMBER_OF_EDGES: 3
HAS_SELF_LOOPS: False
EDGE_LIST: [["A", "B", 1], ["B", "C", 3], ["C", "A", 2]]
VERTEX_DEGREES: {{"A": {{"in_degree": 1, "out_degree": 1}}, "B": {{"in_degree": 1, "out_degree": 1}}, "C": {{"in_degree": 1, "out_degree": 1}}}}

Important:
- Use Python literal formatting for all structured values.
- Boolean values must be written as True or False.
- EDGE_LIST must be a list of [source, destination, weight] triples.
- VERTEX_DEGREES must map each vertex to a dictionary with in_degree and out_degree keys.
- For undirected graphs, in_degree and out_degree should be equal to the undirected degree.
- Sort the EDGE_LIST and VERTEX_DEGREES entries in a deterministic vertex order before writing them.
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
        
        directed_folders = sorted([d for d in os.listdir(DIRECTED_GRAPHS_BASE)
                                   if os.path.isdir(os.path.join(DIRECTED_GRAPHS_BASE, d))])
        
        undirected_folders = sorted([d for d in os.listdir(UNDIRECTED_GRAPHS_BASE)
                                     if os.path.isdir(os.path.join(UNDIRECTED_GRAPHS_BASE, d))])
        
        print(f"  Found {len(directed_folders)} directed graph folders")
        print(f"  Found {len(undirected_folders)} undirected graph folders")
        
        directed_count = None if NUMBER_OF_GRAPHS is None else NUMBER_OF_GRAPHS // 2
        undirected_count = None if NUMBER_OF_GRAPHS is None else NUMBER_OF_GRAPHS // 2
        
        selected_directed = select_graphs(directed_folders, directed_count)
        selected_undirected = select_graphs(undirected_folders, undirected_count)
        
        selected_graphs = [
            ('directed', d) for d in selected_directed
        ] + [
            ('undirected', u) for u in selected_undirected
        ]
        
        print(f"\n  Selected graphs:")
        for graph_type, folder in selected_graphs:
            print(f"    - {graph_type}/{folder}")
            
        combinations = []
        for graph_type, graph_folder in selected_graphs:
            base_dir = DIRECTED_GRAPHS_BASE if graph_type == 'directed' else UNDIRECTED_GRAPHS_BASE
            combinations.append({
                'graph_type': graph_type,
                'graph_folder': graph_folder,
                'base_dir': base_dir,
            })
            
        total_combinations = len(combinations)
        print(f"\n  Total graph combinations: {total_combinations}")
        print(f"  MAX_PROMPTS_COUNT: {MAX_PROMPTS_COUNT if MAX_PROMPTS_COUNT else 'None (all)'}")
        
        if MAX_PROMPTS_COUNT:
            combinations = random.sample(
                combinations,
                min(MAX_PROMPTS_COUNT, len(combinations))
            )
            print(f"  Limited to: {len(combinations)} graphs")
            
        start_idx = 0
        
    print("\n[STEP 2] Processing graphs...\n")
    
    for idx, combo in enumerate(combinations[start_idx:], start_idx + 1):
        graph_type = combo['graph_type']
        graph_folder = combo['graph_folder']
        base_dir = combo['base_dir']
        
        graph_filepath = os.path.join(base_dir, graph_folder, f"graph_{graph_folder}.txt")
        
        print(f"[{idx}/{len(combinations)}] {graph_type}/{graph_folder}")
        
        graph = Graph.from_file(graph_filepath, directed=(graph_type == "directed"))
        
        if USE_TWINPORT:
            graph_description = graph.twinport_description()
        else:
            graph_description = graph.incident_description()
        
        if not graph_description:
            print(f"  ❌ ERROR: Failed to load graph")
            save_to_log(f"\n[{idx}] {graph_type}/{graph_folder}")
            save_to_log(f"  ❌ ERROR: Failed to load files\n")
            continue
        
        prompt = GRAPH_PROMPT.format(
            graph_description=graph_description,
        )

        print(prompt) # just print the raw prompt
        
        for attempt in range(1, NUMBER_OF_PROMPTS_PER_GRAPH + 1):
            print(f"\n--- Attempt {attempt} ---")
            start_time = time.time()
            response = call_model(prompt)
            end_time = time.time()
            duration = round(end_time - start_time, 2)

            print("LLM's Complete, Unedited Response:", response)
            print("-" * 80)

            save_to_log(f"\\n[{idx} - Attempt {attempt}] {graph_type}/{graph_folder}")

            csv_row = {
                "Attempt": attempt,
                "Graph Type": graph_type,
                "Graph Folder": graph_folder,
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
            "next_idx": idx
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=4)
    
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    print("\n✅ SLM Inference completed.")


def run_graph_structure_judge():
    print("="*80)
    print("GRAPH STRUCTURE JUDGE")
    print("="*80)
    
    import pandas as pd
    
    if not os.path.exists(LLM_CSV_LOG_FILE):
        print(f"❌ CSV log file not found at {LLM_CSV_LOG_FILE}. Run slm inference first.")
        return
        
    df = pd.read_csv(LLM_CSV_LOG_FILE)
    
    # Ensure columns exist
    for col in ["Adjacency Match", "Adjacency Error", "Adjacency Entry Match Score", "Adjacency Row Match Score", 
                "LLM Adjacency Matrix", "Expected Adjacency Matrix", "Metadata Match", "LLM Metadata", 
                "Expected Metadata", "Metadata Error Reason"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype='object')
            
    stats = {
        'total': 0,
        'adjacency_correct': 0,
        'adjacency_wrong': 0,
        'adjacency_error': 0,
        'metadata_correct': 0,
        'metadata_wrong': 0,
        'metadata_error': 0,
        'failed_adjacency_matrices': [],
    }
    
    total_rows = len(df)
    
    for index, row in df.iterrows():
        stats['total'] += 1
        
        current_match = row.get('Metadata Match', None)
        if pd.notna(current_match) and str(current_match).strip() != "":
            # Existing score found
            try:
                if str(row.get('Adjacency Match', '')).lower() == 'true':
                    stats['adjacency_correct'] += 1
                elif str(row.get('Adjacency Error', '')).lower() == 'true':
                    stats['adjacency_error'] += 1
                else:
                    stats['adjacency_wrong'] += 1
                    
                if str(row.get('Metadata Match', '')).lower() == 'true':
                    stats['metadata_correct'] += 1
                elif str(row.get('Metadata Match', '')).lower() == 'false':
                    stats['metadata_wrong'] += 1
                else:
                    stats['metadata_error'] += 1
            except:
                pass
            continue
            
        print(f"Judging row {index+1}/{total_rows}: {row['Graph Type']}/{row['Graph Folder']} Attempt {row['Attempt']}")
        
        graph_type = row['Graph Type']
        # Convert graph_folder to 2-digit string to handle pandas auto-conversion to int
        graph_folder = str(row['Graph Folder']).zfill(2)
        base_dir = DIRECTED_GRAPHS_BASE if graph_type == 'directed' else UNDIRECTED_GRAPHS_BASE
        graph_filepath = os.path.join(base_dir, graph_folder, f"graph_{graph_folder}.txt")
        
        try:
            graph = Graph.from_file(graph_filepath, directed=(graph_type == "directed"))
        except Exception as e:
            print(f"  ❌ ERROR loading graph: {e}")
            stats['metadata_error'] += 1
            stats['adjacency_error'] += 1
            continue

        expected_matrix = graph.get_adjacency_matrix()
        response = row['Raw Response']
        
        extracted_matrix = extract_adjacency_matrix_from_response(response)
        
        adjacency_match = False
        adjacency_error = False
        if expected_matrix is None:
            adjacency_error = True
            stats['adjacency_error'] += 1
        elif extracted_matrix is None:
            stats['adjacency_wrong'] += 1
            stats['failed_adjacency_matrices'].append(f"{graph_type}/{graph_folder}")
        elif graph.adjacency_matrix_matches(extracted_matrix):
            adjacency_match = True
            stats['adjacency_correct'] += 1
        else:
            stats['adjacency_wrong'] += 1

        df.at[index, 'Adjacency Match'] = adjacency_match
        df.at[index, 'Adjacency Error'] = adjacency_error
        df.at[index, 'Adjacency Entry Match Score'] = graph.get_adjacency_matrix_entry_score(extracted_matrix) if extracted_matrix is not None else 0.0
        df.at[index, 'Adjacency Row Match Score'] = graph.get_adjacency_matrix_row_score(extracted_matrix) if extracted_matrix is not None else 0.0
        df.at[index, 'LLM Adjacency Matrix'] = matrix_to_string(extracted_matrix)
        df.at[index, 'Expected Adjacency Matrix'] = matrix_to_string(expected_matrix)

        metadata_ok, extracted_metadata, expected_metadata, reason = graph_metadata_matches(response, graph)
        if metadata_ok:
            stats['metadata_correct'] += 1
        else:
            stats['metadata_wrong'] += 1

        df.at[index, 'Metadata Match'] = metadata_ok
        df.at[index, 'LLM Metadata'] = json.dumps(extracted_metadata) if extracted_metadata else ""
        df.at[index, 'Expected Metadata'] = json.dumps(expected_metadata) if expected_metadata else ""
        df.at[index, 'Metadata Error Reason'] = reason if not metadata_ok else ""
        
        df.to_csv(LLM_CSV_LOG_FILE, index=False)

    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    final_stats_text = f"""
FINAL STATISTICS
================

Total Graphs Processed: {stats['total']}

ADJACENCY MATRIX:
  ✅ Correct: {stats['adjacency_correct']} ({100*stats['adjacency_correct']//stats['total'] if stats['total'] > 0 else 0}%)
  ❌ Wrong: {stats['adjacency_wrong']}
  ⚠️  Error: {stats['adjacency_error']}

GRAPH METADATA:
  ✅ Correct: {stats['metadata_correct']} ({100*stats['metadata_correct']//stats['total'] if stats['total'] > 0 else 0}%)
  ❌ Wrong: {stats['metadata_wrong']}
  ⚠️  Error: {stats['metadata_error']}
"""
    
    if stats.get('failed_adjacency_matrices'):
        final_stats_text += "\nFAILED ADJACENCY MATRICES (Returned None):\n"
        for failed in stats['failed_adjacency_matrices']:
            final_stats_text += f"  - {failed}\n"
            
    print(final_stats_text)
    save_to_log(final_stats_text)
    print(f"\nLog file: {LLM_LOG_FILE}")
    save_to_log(f"\n{'='*80}\n")
    
if __name__ == "__main__":
    # Comment out either function as needed
    run_slm_inference()
    run_graph_structure_judge()