"""
Tic-Tac-Toe Heuristic Evaluator
================================

This script batch-evaluates a folder of externally-supplied heuristic
functions against a strong built-in heuristic, 2,500 games each
(500 games at each of 5 evaluator accuracy levels), and prints
each test taker's scores in a table.

BOARD INTERFACE CONTRACT
-------------------------
A board is a 3x3 list of lists (rows, then columns), e.g.:

    board[0] = [row0col0, row0col1, row0col2]
    board[1] = [row1col0, row1col1, row1col2]
    board[2] = [row2col0, row2col1, row2col2]

Each cell holds one of:
    ''   -> empty cell
    'X'  -> X has been placed there
    'Y'  -> Y has been placed there

FIXED SYMBOL ASSIGNMENT
-------------------------
In this evaluation harness, the test taker's heuristic is always assigned
'X' and the built-in evaluator heuristic is always assigned 'Y' -- this
never changes, no matter who moves first in a given match. What does
alternate across matches (for fairness) is which symbol gets the opening
move: in some games 'X' moves first, in others 'Y' does. A heuristic
function is only ever invoked when it is actually its own symbol's turn,
so it never needs to figure out "which side am I" -- the test taker's
function can simply assume it is always choosing X's move.

HEURISTIC FUNCTION CONTRACT
----------------------------
A heuristic function has the signature:

    def myHeuristic(board: list[list[str]]) -> tuple[int, int]

It receives a snapshot (a deep copy) of the current board and must return
a tuple (row, col) with 0 <= row < 3 and 0 <= col < 3, identifying an
empty cell to play in.

REQUIREMENTS TO RUN
--------------------
Create a folder named `heuristics` next to this script. Inside it, place
one .py file per test taker, named after them (spaces and hyphens are
fine, e.g. "Jane Doe.py" or "Anna-Marie O'Brien.py"). Each file must
define a function named exactly `myHeuristic`:

    def myHeuristic(board):
        ...
        return (row, col)

SCORING
-------
For each test taker's file, 2,500 games are played against the built-in
heuristic -- 500 games at each of 5 evaluator accuracy levels:
  50%, 65%, 80%, 95%, and 100%.

"Accuracy" here means the probability that the evaluator plays its
optimal (minimax) move; with probability (1 - accuracy) it plays a
uniformly random legal move instead.

The test taker is always 'X' and the built-in heuristic is always 'Y';
which one opens the game alternates match to match. Their score at each
accuracy level is:

    score = (number of wins + draws) * 0.2

OUTPUT
------
For each heuristic file, a single-line progress bar is shown while its
2,500 games are played. Once every file has been evaluated, a final table
of scores at each accuracy level -- plus a grand total -- is printed.
"""

import copy
import glob
import importlib.util
import os
import random
import sys

# ---------------------------------------------------------------------------
# Board / game primitives
# ---------------------------------------------------------------------------

EMPTY = ''
PLAYER_X = 'X'
PLAYER_Y = 'Y'

MATCHES_PER_LEVEL = 500
SCORE_MULTIPLIER = 0.2
HEURISTICS_DIR = 'heuristics'

# Evaluator accuracy levels: probability of playing the optimal move.
ACCURACY_LEVELS = [0.50, 0.65, 0.80, 0.95, 1.00]
TOTAL_MATCHES = MATCHES_PER_LEVEL * len(ACCURACY_LEVELS)  # 2500


def create_board():
    return [[EMPTY, EMPTY, EMPTY] for _ in range(3)]


def get_valid_moves(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == EMPTY]


def is_valid_move(board, move):
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        return False
    r, c = move
    if not (isinstance(r, int) and isinstance(c, int)):
        return False
    if not (0 <= r < 3 and 0 <= c < 3):
        return False
    return board[r][c] == EMPTY


def check_winner(board):
    lines = []
    for i in range(3):
        lines.append([board[i][0], board[i][1], board[i][2]])          # rows
        lines.append([board[0][i], board[1][i], board[2][i]])          # cols
    lines.append([board[0][0], board[1][1], board[2][2]])              # diag
    lines.append([board[0][2], board[1][1], board[2][0]])              # anti-diag

    for line in lines:
        if line[0] != EMPTY and line[0] == line[1] == line[2]:
            return line[0]
    return None


def is_draw(board):
    return check_winner(board) is None and len(get_valid_moves(board)) == 0


def current_player(board, starting_symbol):
    """
    Whoever moved first (`starting_symbol`) is to move again whenever the
    total number of moves so far is even; otherwise it's the other side's
    turn. This correctly handles matches where 'Y' opens instead of 'X'.
    """
    total_moves = sum(row.count(PLAYER_X) + row.count(PLAYER_Y) for row in board)
    return starting_symbol if total_moves % 2 == 0 else other_player(starting_symbol)


def other_player(p):
    return PLAYER_Y if p == PLAYER_X else PLAYER_X


# ---------------------------------------------------------------------------
# Strong built-in heuristic: minimax with alpha-beta pruning + variable randomness
# ---------------------------------------------------------------------------

def _minimax(board, player, maximizing_player, alpha=-float('inf'), beta=float('inf')):
    """
    Returns (score, move) for `player` to move, from the perspective of
    `maximizing_player` (score > 0 favors maximizing_player winning,
    score < 0 favors the opponent, 0 is a draw).
    """
    winner = check_winner(board)
    if winner == maximizing_player:
        return 1, None
    if winner == other_player(maximizing_player):
        return -1, None
    if is_draw(board):
        return 0, None

    moves = get_valid_moves(board)
    best_move = moves[0]

    if player == maximizing_player:
        best_score = -float('inf')
        for move in moves:
            r, c = move
            board[r][c] = player
            score, _ = _minimax(board, other_player(player), maximizing_player, alpha, beta)
            board[r][c] = EMPTY
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score, best_move
    else:
        best_score = float('inf')
        for move in moves:
            r, c = move
            board[r][c] = player
            score, _ = _minimax(board, other_player(player), maximizing_player, alpha, beta)
            board[r][c] = EMPTY
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score, best_move


def make_best_heuristic(accuracy):
    """
    Returns a best_heuristic function that plays the optimal minimax move
    with probability `accuracy`, and a uniformly random legal move otherwise.

    `accuracy` should be a float in [0.0, 1.0].
      - 1.00 -> always plays the optimal move (hardest)
      - 0.50 -> plays optimally only half the time (as in the original script)
    """
    def best_heuristic(board):
        moves = get_valid_moves(board)
        if not moves:
            return None

        _, move = _minimax(copy.deepcopy(board), PLAYER_Y, PLAYER_Y)
        if move is None:
            move = random.choice(moves)

        # With probability (1 - accuracy), override with a random move.
        if random.random() >= accuracy:
            move = random.choice(moves)

        return move

    return best_heuristic


# ---------------------------------------------------------------------------
# Match / tournament logic
# ---------------------------------------------------------------------------

def _safe_call(heuristic_func, board):
    """Calls a heuristic defensively; any exception or bad output -> None."""
    try:
        move = heuristic_func(copy.deepcopy(board))
    except Exception:
        return None
    return move


def play_match(test_taker_heuristic, evaluator_heuristic, starting_symbol):
    """
    Plays a single game. The test taker's heuristic always controls 'X'
    and the built-in evaluator_heuristic always controls 'Y' -- this mapping
    never changes. `starting_symbol` ('X' or 'Y') decides who opens this
    particular match. Returns 'X', 'Y', or 'draw'.
    A player that returns an invalid move forfeits the match.
    """
    board = create_board()
    controllers = {PLAYER_X: test_taker_heuristic, PLAYER_Y: evaluator_heuristic}

    while True:
        winner = check_winner(board)
        if winner is not None:
            return winner
        if is_draw(board):
            return 'draw'

        player = current_player(board, starting_symbol)
        move = _safe_call(controllers[player], board)

        if not is_valid_move(board, move):
            # Invalid / missing move -> forfeit
            return other_player(player)

        r, c = move
        board[r][c] = player


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def print_progress(label, current, total, bar_length=30):
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = '#' * filled + '-' * (bar_length - filled)
    sys.stdout.write(f'\r{label}: [{bar}] {current}/{total}')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')


# ---------------------------------------------------------------------------
# Heuristic file discovery / loading
# ---------------------------------------------------------------------------

def discover_heuristic_files(directory):
    """Returns a sorted list of .py file paths inside `directory`."""
    pattern = os.path.join(directory, '*.py')
    return sorted(glob.glob(pattern))


def load_heuristic_from_file(filepath):
    """
    Dynamically imports the given .py file (whose name may contain spaces
    or hyphens, so it can't be imported normally) and returns its
    myHeuristic function.
    """
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, 'myHeuristic'):
        raise AttributeError("does not define a function named 'myHeuristic'")

    return module.myHeuristic


# ---------------------------------------------------------------------------
# Evaluation of a single test taker's heuristic
# ---------------------------------------------------------------------------

def evaluate_heuristic(name, heuristic_func):
    """
    Plays MATCHES_PER_LEVEL games at each accuracy level in ACCURACY_LEVELS
    (TOTAL_MATCHES games in total) of heuristic_func (always 'X') vs. the
    evaluator (always 'Y'). Returns a list of (wins, draws, losses) tuples,
    one per accuracy level.
    """
    level_results = []
    match_index = 0

    for accuracy in ACCURACY_LEVELS:
        evaluator = make_best_heuristic(accuracy)
        wins = 0
        draws = 0
        losses = 0

        for i in range(1, MATCHES_PER_LEVEL + 1):
            match_index += 1
            # Alternate who opens each match for fairness.
            starting_symbol = PLAYER_X if i % 2 == 1 else PLAYER_Y
            result = play_match(heuristic_func, evaluator, starting_symbol)

            if result == PLAYER_X:
                wins += 1
            elif result == 'draw':
                draws += 1
            else:
                losses += 1

            print_progress(name, match_index, TOTAL_MATCHES)

        level_results.append((wins, draws, losses))

    return level_results


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_results_table(results):
    """
    Prints a formatted table with per-accuracy-level W/D/L breakdown and a
    grand total score.

    `results` is a list of (name, level_results) where level_results is a
    list of (wins, draws, losses) tuples aligned with ACCURACY_LEVELS.
    """
    # Column headers
    level_headers = [f"{int(a * 100)}%" for a in ACCURACY_LEVELS]
    col_name  = "Name"
    col_total = "Total"
    wdl_label = "W/D/L"

    # Pre-compute display data
    display = []
    for name, level_results in results:
        wdl_strings = []
        total_wins_draws = 0
        for wins, draws, losses in level_results:
            wdl_strings.append(f"{wins}/{draws}/{losses}")
            total_wins_draws += wins + draws
        total_score = round(total_wins_draws * SCORE_MULTIPLIER, 1)
        display.append((name, wdl_strings, total_score))

    # Column widths
    name_width = max(len(col_name), max(len(d[0]) for d in display))
    wdl_width  = max(len(wdl_label), max(
        max(len(s) for s in d[1]) for d in display
    ))
    total_width = max(len(col_total), max(len(str(d[2])) for d in display))

    # Header row
    header_cells = [col_name.ljust(name_width)]
    for h in level_headers:
        header_cells.append(h.center(wdl_width))
    header_cells.append(col_total.center(total_width))
    header = " | ".join(header_cells)

    # Sub-header (W/D/L labels)
    sub_cells = [" " * name_width]
    for _ in ACCURACY_LEVELS:
        sub_cells.append(wdl_label.center(wdl_width))
    sub_cells.append(" " * total_width)
    sub_header = " | ".join(sub_cells)

    separator = "-+-".join([
        "-" * name_width,
        *["-" * wdl_width for _ in ACCURACY_LEVELS],
        "-" * total_width,
    ])

    print()
    print(header)
    print(sub_header)
    print(separator)

    for name, wdl_strings, total_score in display:
        row_cells = [name.ljust(name_width)]
        for s in wdl_strings:
            row_cells.append(s.center(wdl_width))
        row_cells.append(str(total_score).center(total_width))
        print(" | ".join(row_cells))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(HEURISTICS_DIR):
        sys.stderr.write(f"Directory '{HEURISTICS_DIR}' not found.\n")
        sys.exit(1)

    filepaths = discover_heuristic_files(HEURISTICS_DIR)
    if not filepaths:
        sys.stderr.write(f"No .py files found in '{HEURISTICS_DIR}'.\n")
        sys.exit(1)

    results = []

    for filepath in filepaths:
        name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            heuristic_func = load_heuristic_from_file(filepath)
        except Exception as exc:
            sys.stderr.write(f"Skipping '{name}': {exc}\n")
            continue

        level_results = evaluate_heuristic(name, heuristic_func)
        results.append((name, level_results))

    print_results_table(results)


if __name__ == '__main__':
    main()
