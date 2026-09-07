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
USE_SLM = True  # Set to True to use local SLM (via LM Studio), False to use Gemini via agy CLI
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_MODEL = "google/gemma-3-4b"
SLM_TEMPERATURE = 0.3

LLM = "gemini-3.8-flash-high"

# Set to None to process all combinations, or an integer to limit prompts
MAX_PROMPTS_COUNT = None  # Change to 4, 10, etc. to limit
PROBLEMS_FROM_EACH_CATEGORY = None # if set to None, then select ALL problems
NUMBER_OF_PROMPTS_PER_PROBLEM_SCENARIO = 3 # the LLM or SLM will be tested with the exact same problem scenario this many times (API calls are memoryless anyway)

# Select which types of problems to test
# 'standard' (only problemX.txt), 'distractor' (only problemX_distractor.txt), or 'all'
PROBLEM_TYPE_FILTER = 'all'

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_BASE = os.path.join(BASE_DIR, "Problems")
LLM_LOG_FILE = os.path.join(BASE_DIR, "llm_log_problem.txt")
LLM_CSV_LOG_FILE = os.path.join(BASE_DIR, "llm_log_problem.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "progress_problem.json")

# ============================================================================
# LLM / SLM INFERENCE HELPER
# ============================================================================

def call_model(prompt: str) -> str:
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
                "seed": random.randint(1, 10000000),
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
                LLM,
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

# A taxonomy of algorithm properties to guide the LLM judge in scoring similarity
GRAPH_ALGORITHMS_TAXONOMY = {
    "Dijkstra": {"type": "Shortest Path", "weights": "Non-negative only", "traversal": "Greedy"},
    "Bellman-Ford": {"type": "Shortest Path", "weights": "Handles negative weights", "traversal": "Dynamic Programming"},
    "BFS": {"type": "Traversal/Unweighted Shortest Path", "weights": "Unweighted", "traversal": "Level-order"},
    "DFS": {"type": "Traversal/Cycle Detection", "weights": "Unweighted", "traversal": "Depth-first"},
    "Floyd-Warshall": {"type": "All Pairs Shortest Path", "weights": "Handles negative weights", "traversal": "Dynamic Programming"},
    "Kruskal's Algorithm": {"type": "Minimum Spanning Tree", "weights": "Weighted", "traversal": "Greedy (Edge-based)"},
    "Prim's Algorithm": {"type": "Minimum Spanning Tree", "weights": "Weighted", "traversal": "Greedy (Node-based)"},
    "Ford-Fulkerson": {"type": "Maximum Flow", "weights": "Capacities", "traversal": "Augmenting Paths"},
    "Edmonds-Karp": {"type": "Maximum Flow", "weights": "Capacities", "traversal": "BFS Augmenting Paths"},
    "Tarjan's SCC": {"type": "Strongly Connected Components", "weights": "Unweighted", "traversal": "DFS based"},
    "Kosaraju's Algorithm": {"type": "Strongly Connected Components", "weights": "Unweighted", "traversal": "Two-pass DFS"},
    "Kahn's Algorithm": {"type": "Topological Sort", "weights": "Unweighted", "traversal": "In-degree based"},
    "Hopcroft-Karp": {"type": "Bipartite Matching", "weights": "Unweighted", "traversal": "BFS+DFS"},
    "Held-Karp": {"type": "TSP", "weights": "Weighted", "traversal": "Dynamic Programming"},
}

# Prompt template
PROBLEM_PROMPT = """DO NOT USE THE INTERNET.

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
    # clear the csv as well:
    with open(LLM_CSV_LOG_FILE, "w", encoding="utf-8") as f:
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

def grade_algorithm_choice(algo_name, explanation, problem_category):
    """
    Grade the LLM's algorithm name against known valid algorithms for the
    given problem category, using LLM as a judge.
    Returns: (score, matched_name, justification)
      score: int between 0 and 100
    """
    if not algo_name:
        return 0, None, "No algorithm extracted"

    valid_algos = GRAPH_ALGORITHMS.get(problem_category, [])
    if not valid_algos:
        return 0, None, "No valid algorithms defined for this category"

    prompt = f'''
You are an expert computer science grader evaluating an LLM's proposed graph algorithm for a specific problem.
Problem Category: {problem_category}
Valid Algorithms (Gold Standard): {valid_algos}

Proposed Algorithm from LLM: {algo_name}
LLM's Explanation: {explanation}

Algorithm Taxonomy Reference:
{json.dumps(GRAPH_ALGORITHMS_TAXONOMY, indent=2)}

Compute a graded similarity score between the Proposed Algorithm and the closest Valid Algorithm.
Score Guidelines:
- 100: Exact match or functionally identical equivalent (e.g., "Breadth First Search" for "BFS").
- 80: Very similar (e.g., proposing Bellman-Ford instead of Dijkstra; it handles weights but is less optimal).
- 40: Shared domain but fundamentally different constraints (e.g., proposing BFS instead of Dijkstra; fails on weighted graphs).
- 0: Completely unrelated (e.g., DFS for Max Flow).

Reply ONLY with a valid JSON object in the following format. Do not include markdown blocks or any other text.
{{
    "Score": <integer between 0 and 100>,
    "Matched_Algorithm": "<The closest algorithm from the Valid Algorithms list>",
    "Justification": "<Brief 10-30 word explanation of the score based on shared or missing properties>"
}}
'''
    import subprocess
    try:
        result = subprocess.run(
            [
                r"C:\Users\ASUS\AppData\Local\agy\bin\agy.exe",
                "--model",
                "gemini-3.8-flash-high",
                "-p",
                prompt,
            ],
            capture_output=True,
            text=True,
        )
        response = result.stdout.strip()
        if response.startswith("```json"):
            response = response.replace("```json", "", 1).strip()
        if response.endswith("```"):
            response = response[:-3].strip()
            
        data = json.loads(response)
        return int(data.get("Score", 0)), data.get("Matched_Algorithm", ""), data.get("Justification", "")
            
    except Exception as e:
        print(f"Error during LLM verification: {e}")
        
    return 0, None, "LLM Judge Failed"

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
            all_txt_files = [f for f in os.listdir(category_path) if f.endswith('.txt')]
            
            if PROBLEM_TYPE_FILTER == 'standard':
                problem_files = sorted([f for f in all_txt_files if not f.endswith('_distractor.txt')])
            elif PROBLEM_TYPE_FILTER == 'distractor':
                problem_files = sorted([f for f in all_txt_files if f.endswith('_distractor.txt')])
            else:
                problem_files = sorted(all_txt_files)
            
            if PROBLEMS_FROM_EACH_CATEGORY is None:
                selected = problem_files
            else:
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
            'perfect_matches': 0, # Score == 100
            'partial_matches': 0, # 0 < Score < 100
            'misses': 0,          # Score == 0
            'total_score': 0.0,   # Accumulate all scores for average
            'algorithm_error': 0,
        }
        start_idx = 0
        
    print("\n[STEP 2] Processing problems...\n")
    
    for idx, combo in enumerate(combinations[start_idx:], start_idx + 1):
        problem_category = combo['problem_category']
        problem_path = combo['problem_path']
        
        problem_name = os.path.basename(problem_path)
        
        print(f"[{idx}/{len(combinations)}] {problem_category}/{problem_name}")
        
        problem_description = load_file(problem_path)
        
        if not problem_description:
            print(f"  ❌ ERROR: Failed to load problem")
            save_to_log(f"\n[{idx}] {problem_category}/{problem_name}")
            save_to_log(f"  ❌ ERROR: Failed to load files\n")
            continue
        
        prompt = PROBLEM_PROMPT.format(
            problem_description=problem_description,
        )

        print(prompt) # just print the raw prompt
        
        for attempt in range(1, NUMBER_OF_PROMPTS_PER_PROBLEM_SCENARIO + 1):
            print(f"\n--- Attempt {attempt} ---")
            stats['total'] += 1
            start_time = time.time()
            response = call_model(prompt)
            end_time = time.time()
            duration = round(end_time - start_time, 2)

            print("LLM's Complete, Unedited Response:", response)
            print("-" * 80)

            extracted_algo = extract_algorithm_from_response(response)
            extracted_explanation = extract_explanation_from_response(response)

            match = False
            matched_name = ""
            valid_algos_for_category = GRAPH_ALGORITHMS.get(problem_category, [])

            score, matched_name, justification = grade_algorithm_choice(extracted_algo, extracted_explanation, problem_category)
            
            # Update comprehensive stats tracking
            stats['total_score'] += score
            if score == 100:
                stats['perfect_matches'] += 1
            elif score > 0:
                stats['partial_matches'] += 1
            else:
                stats['misses'] += 1
                
            print(f"  => Score: {score} | Matched: {matched_name}")
            print(f"  => Justification: {justification}")
            
            save_to_log(f"\\n[{idx} - Attempt {attempt}] {problem_category}/{problem_name}")
            save_to_log(f"  ALGORITHM CHOICE SCORE: {score}\\n")
            save_to_log(f"  LLM choice: {extracted_algo}\\n")
            save_to_log(f"  Closest match: {matched_name}\\n")
            save_to_log(f"  Justification: {justification}\\n")
            save_to_log(f"  Valid algorithms: {valid_algos_for_category}\\n")


            csv_row = {
                "Attempt": attempt,
                "Problem Category": problem_category,
                "Problem Name": problem_name,
                "Is Distractor": "Yes" if "distractor" in problem_name.lower() else "No",
                "Execution Time (s)": duration,
                "LLM Algorithm Choice": extracted_algo if extracted_algo else "",
                "Valid Algorithms": json.dumps(valid_algos_for_category),
                "LLM Explanation": extracted_explanation if extracted_explanation else "",
                "Algorithm Match Score": score,
                "Matched Algorithm Name": matched_name if matched_name else "",
                "Judge Justification": justification,          "Prompt Length (chars)": len(prompt),
                "Response Length (chars)": len(response),
                "Raw Response": response
            }
            fieldnames = list(csv_row.keys())
            log_to_csv(LLM_CSV_LOG_FILE, csv_row, fieldnames)

            avg_score = stats['total_score'] / stats['total'] if stats['total'] > 0 else 0
            print(f"    Average Score: {avg_score:.2f} | Perfects: {stats['perfect_matches']} | Partials: {stats['partial_matches']} | Misses: {stats['misses']}")
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
    
    avg_score = stats['total_score'] / stats['total'] if stats['total'] > 0 else 0
    final_stats_text = f"""
FINAL STATISTICS
================

Total Attempts Evaluated: {stats['total']}

GRADED ALGORITHM CHOICE METRICS:
  🏆 Average Similarity Score: {avg_score:.2f} out of 100
  ✅ Perfect Matches (100): {stats['perfect_matches']}
  ⚠️  Partial Matches (40 or 80): {stats['partial_matches']}
  ❌ Total Misses (0): {stats['misses']}
"""
            
    print(final_stats_text)
    save_to_log(final_stats_text)
    print(f"\nLog file: {LLM_LOG_FILE}")
    save_to_log(f"\n{'='*80}\n")
    
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

if __name__ == "__main__":
    main()
