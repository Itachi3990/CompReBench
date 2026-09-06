import os
import random
import re
import json
import sys
import csv
import time

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

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_BASE = os.path.join(BASE_DIR, "Problems")
LLM_LOG_FILE = os.path.join(BASE_DIR, "llm_log_problem.txt")
LLM_CSV_LOG_FILE = os.path.join(BASE_DIR, "llm_log_problem.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "prompt_generation_progress_problem.json")

# ============================================================================
# LLM / SLM INFERENCE HELPER
# ============================================================================

def call_model(prompt: str, max_tokens: int = 2048) -> str:
    """
    Unified model invocation function.
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

# Prompt template
GRAPH_PROMPT = """DO NOT USE THE INTERNET.

Consider the following problem scenario that needs to be mapped to a graph problem:

{problem_description}

Decide which algorithm(s) you would use to solve this problem. Do NOT solve the problem itself.

Then provide the exact algorithm answer below. Use deterministic formatting and do not add extra labels or commentary.

Your output must EXACTLY follow this format, in this exact order:

ALGORITHM NAME: <ALGORITHM NAME>
EXPLANATION OF ALGORITHM CHOICE: <EXPLANATION>

Important:
- The fields ALGORITHM NAME and EXPLANATION OF ALGORITHM CHOICE are mandatory and must remain in the output. Whatever algorithm(s) you choose, clearly state the algorithm name(s) and provide a brief explanation (10-30 words) of why that algorithm is appropriate for this problem.
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

def log_to_csv(filepath, row_dict, fieldnames):
    file_exists = os.path.isfile(filepath)
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

def extract_algorithm_from_response(response):
    """Extract algorithm name from LLM response."""
    match = re.search(r"ALGORITHM NAME:\s*(.+?)(?:\n|$)", response)
    if match:
        return match.group(1).strip()
    return None

def extract_explanation_from_response(response):
    """Extract explanation from LLM response."""
    match = re.search(r"EXPLANATION OF ALGORITHM CHOICE:\s*(.+?)(?:\n|$)", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

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
            for valid in valid_algos:
                if matched.lower() == valid.lower():
                    return True, valid
            return True, matched
        elif response.startswith("YES"):
            return True, algo_name
            
    except Exception as e:
        print(f"Error during LLM verification: {e}")
        
    return False, None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("PROBLEM SCENARIO TESTER")
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
            'algorithm_correct': 0,
            'algorithm_wrong': 0,
            'algorithm_error': 0,
        })
        start_idx = state.get("next_idx", 0)
        
        save_to_log("="*80 + "\n")
        save_to_log(f"RESUMING PROBLEM SCENARIO TESTER LOG (from index {start_idx+1})\n")
        save_to_log("="*80 + "\n\n")
    else:
        clear_log()
        save_to_log("="*80 + "\n")
        save_to_log("PROBLEM SCENARIO TESTER LOG\n")
        save_to_log("="*80 + "\n\n")
        
        print("\n[STEP 1] Collecting problems...")
        
        problem_categories = [d for d in os.listdir(PROBLEMS_BASE)
                             if os.path.isdir(os.path.join(PROBLEMS_BASE, d))]
        
        selected_problems = {}
        
        for category in problem_categories:
            category_path = os.path.join(PROBLEMS_BASE, category)
            problem_files = sorted([f for f in os.listdir(category_path)
                                   if f.endswith('.txt')])
            
            selected = random.sample(problem_files, min(PROBLEMS_FROM_EACH_CATEGORY, len(problem_files)))
            selected_problems[category] = [
                os.path.join(category_path, f) for f in selected
            ]
            
            print(f"  {category}: selected {len(selected)} of {len(problem_files)} problems")
        
        total_problems = sum(len(v) for v in selected_problems.values())
        print(f"\n  Total problems selected: {total_problems}")
        
        combinations = []
        for category, problem_paths in selected_problems.items():
            for problem_path in problem_paths:
                combinations.append({
                    'problem_category': category,
                    'problem_path': problem_path,
                })
        
        print(f"  Total combinations: {len(combinations)}")
        print(f"  MAX_PROMPTS_COUNT: {MAX_PROMPTS_COUNT if MAX_PROMPTS_COUNT else 'None (all)'}")
        
        if MAX_PROMPTS_COUNT:
            combinations = random.sample(
                combinations,
                min(MAX_PROMPTS_COUNT, len(combinations))
            )
            print(f"  Limited to: {len(combinations)} combinations")
            
        stats = {
            'total': 0,
            'algorithm_correct': 0,
            'algorithm_wrong': 0,
            'algorithm_error': 0,
        }
        start_idx = 0
        
    print("\n[STEP 2] Processing problems...\n")
    
    for idx, combo in enumerate(combinations[start_idx:], start_idx + 1):
        problem_category = combo['problem_category']
        problem_path = combo['problem_path']
        
        stats['total'] += 1
        problem_name = os.path.basename(problem_path)
        
        print(f"[{idx}/{len(combinations)}] {problem_category}/{problem_name}")
        
        problem_description = load_file(problem_path)
        
        if not problem_description:
            print(f"  ❌ ERROR: Failed to load problem")
            save_to_log(f"\n[{idx}] {problem_category}/{problem_name}")
            save_to_log(f"  ❌ ERROR: Failed to load files\n")
            continue
        
        prompt = GRAPH_PROMPT.format(
            problem_description=problem_description,
        )

        print(prompt)
        
        start_time = time.time()
        response = call_model(prompt, max_tokens=2048)
        end_time = time.time()
        duration = round(end_time - start_time, 2)

        print("LLM's Complete, Unedited Response:", response)
        print("-" * 80)
        
        extracted_algo = extract_algorithm_from_response(response)
        extracted_explanation = extract_explanation_from_response(response)
        
        match = False
        matched_name = ""
        valid_algos_for_category = GRAPH_ALGORITHMS.get(problem_category, [])

        if extracted_algo is None:
            print(f"  ❌ WRONG ALGORITHM CHOICE (extraction failed)")
            print(f"     LLM algorithm choice: {extracted_algo}")
            print(f"     Valid algorithms for {problem_category}: {valid_algos_for_category}")
            save_to_log(f"\n[{idx}] {problem_category}/{problem_name}")
            save_to_log(f"  ALGORITHM CHOICE: ❌ WRONG (extraction failed)\n")
            save_to_log(f"  LLM algorithm choice: {extracted_algo}\n")
            save_to_log(f"  Valid algorithms for {problem_category}: {valid_algos_for_category}\n")
            stats['algorithm_wrong'] += 1
        else:
            match, matched_name = algorithm_matches_reference(extracted_algo, extracted_explanation, problem_category)
            if match:
                print(f"  ✅ CORRECT ALGORITHM CHOICE ({matched_name})")
                save_to_log(f"\n[{idx}] {problem_category}/{problem_name}")
                save_to_log(f"  ALGORITHM CHOICE: ✅ CORRECT ({matched_name})\n")
                save_to_log(f"  LLM algorithm choice: {extracted_algo}\n")
                save_to_log(f"  Valid algorithms for {problem_category}: {valid_algos_for_category}\n")
                stats['algorithm_correct'] += 1
            else:
                print(f"  ❌ WRONG ALGORITHM CHOICE ({extracted_algo})")
                print(f"     LLM algorithm choice: {extracted_algo}")
                print(f"     Valid algorithms for {problem_category}: {valid_algos_for_category}")
                save_to_log(f"\n[{idx}] {problem_category}/{problem_name}")
                save_to_log(f"  ALGORITHM CHOICE: ❌ WRONG ({extracted_algo})\n")
                save_to_log(f"  LLM algorithm choice: {extracted_algo}\n")
                save_to_log(f"  Valid algorithms for {problem_category}: {valid_algos_for_category}\n")
                stats['algorithm_wrong'] += 1
        
        csv_row = {
            "Problem Category": problem_category,
            "Problem Name": problem_name,
            "Execution Time (s)": duration,
            "Algorithm Match": match,
            "Matched Algorithm Name": matched_name if match else "",
            "LLM Algorithm Choice": extracted_algo if extracted_algo else "",
            "Valid Algorithms": json.dumps(valid_algos_for_category),
            "LLM Explanation": extracted_explanation if extracted_explanation else "",
            "Prompt Length (chars)": len(prompt),
            "Response Length (chars)": len(response),
            "Raw Response": response
        }
        fieldnames = list(csv_row.keys())
        log_to_csv(LLM_CSV_LOG_FILE, csv_row, fieldnames)
        
        print(f"    Algorithm Accuracy:  {stats['algorithm_correct']}/{stats['total']} ({100*stats['algorithm_correct']//stats['total']}%)")
        print()
        
        state_to_save = {
            "combinations": combinations,
            "stats": stats,
            "next_idx": idx
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=4)
    
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    final_stats_text = f"""
FINAL STATISTICS
================

Total Problems Processed: {stats['total']}

ALGORITHM CHOICE:
  ✅ Correct: {stats['algorithm_correct']} ({100*stats['algorithm_correct']//stats['total'] if stats['total'] > 0 else 0}%)
  ❌ Wrong: {stats['algorithm_wrong']}
  ⚠️  Error: {stats['algorithm_error']}
"""
            
    print(final_stats_text)
    save_to_log(final_stats_text)
    print(f"\nLog file: {LLM_LOG_FILE}")
    save_to_log(f"\n{'='*80}\n")
    
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

if __name__ == "__main__":
    main()
