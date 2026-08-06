"""
4x4 Tic-Tac-Toe Heuristic Evaluator
====================================

This script batch-evaluates a folder of externally-supplied heuristic
functions against a strong built-in heuristic, 2,500 games each
(500 games at each of 5 evaluator accuracy levels), and prints
each test taker's scores in a table.

BOARD INTERFACE CONTRACT
-------------------------
A board is a 4x4 list of lists (rows, then columns), e.g.:

    board[0] = [row0col0, row0col1, row0col2, row0col3]
    board[1] = [row1col0, row1col1, row1col2, row1col3]
    board[2] = [row2col0, row2col1, row2col2, row2col3]
    board[3] = [row3col0, row3col1, row3col2, row3col3]

Each cell holds one of:
    ''   -> empty cell
    'X'  -> X has been placed there
    'Y'  -> Y has been placed there

WIN CONDITION
-------------
A player wins by filling an entire row, an entire column, or one of the
two main diagonals (all 4 cells) with their own symbol. Shorter runs (2 or
3 in a row) do not win.

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
a tuple (row, col) with 0 <= row < 4 and 0 <= col < 4, identifying an
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

A NOTE ON THE BUILT-IN HEURISTIC'S SEARCH DEPTH
-------------------------------------------------
On a 3x3 board, brute-force minimax can search all the way to the end of
the game every move. On a 4x4 board (16 cells) that's no longer feasible
-- the full game tree is on the order of 16! nodes. So the built-in
heuristic instead:
  * searches all the way to the end of the game once the board is nearly
    full (<= FULL_SEARCH_THRESHOLD empty cells left), giving perfect play
    in the endgame, and
  * otherwise only looks LOOKAHEAD_DEPTH plies ahead and falls back on a
    static evaluation function that scores how many "live" (unblocked)
    winning lines each side still has, and how many marks they've already
    placed on each.
The random coin is flipped *before* running the search at each accuracy
level, so the (relatively expensive) search is skipped whenever its
result would be discarded anyway.
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

BOARD_SIZE = 4
MATCHES_PER_LEVEL = 500
SCORE_MULTIPLIER = 0.2
HEURISTICS_DIR = 'heuristics'

# Evaluator accuracy levels: probability of playing the optimal move.
ACCURACY_LEVELS = [0.50, 0.65, 0.80, 0.95, 1.00]
TOTAL_MATCHES = MATCHES_PER_LEVEL * len(ACCURACY_LEVELS)  # 2500

# Search-depth tuning for the built-in heuristic (see module docstring).
FULL_SEARCH_THRESHOLD = 8   # <= this many empty cells left -> search to the end of the game
LOOKAHEAD_DEPTH = 4         # otherwise -> only look this many plies ahead


def create_board():
    return [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]


def get_valid_moves(board):
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == EMPTY]


def is_valid_move(board, move):
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        return False
    r, c = move
    if not (isinstance(r, int) and isinstance(c, int)):
        return False
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        return False
    return board[r][c] == EMPTY


def _build_winning_lines():
    lines = []
    for i in range(BOARD_SIZE):
        lines.append([(i, c) for c in range(BOARD_SIZE)])                # row i
        lines.append([(r, i) for r in range(BOARD_SIZE)])                # col i
    lines.append([(i, i) for i in range(BOARD_SIZE)])                    # main diagonal
    lines.append([(i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)])   # anti-diagonal
    return lines


WINNING_LINES = _build_winning_lines()


def _build_cell_lines():
    cell_lines = {(r, c): [] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)}
    for line in WINNING_LINES:
        for (r, c) in line:
            cell_lines[(r, c)].append(line)
    return cell_lines


# Which winning lines pass through each cell -- lets the search check only
# the lines touched by the move that was just played, instead of rescanning
# the whole board after every hypothetical move.
CELL_LINES = _build_cell_lines()


def _build_cell_weights():
    """Positional value = number of winning lines that pass through each cell.
    For a 4x4 board the participation counts are:
        2  3  3  2
        3  4  4  3
        3  4  4  3
        2  3  3  2
    Inner cells (rows 1-2, cols 1-2) each belong to 4 winning lines, while
    corner cells belong to only 2.  Placing a mark on a high-participation
    cell keeps more winning options alive.  Source: elementary combinatorics
    of an N×N board's winning lines (see also trincoll.edu CS notes).
    """
    weights = {(r, c): len(lines) for (r, c), lines in CELL_LINES.items()}
    return weights


CELL_WEIGHTS = _build_cell_weights()


def check_winner(board):
    for line in WINNING_LINES:
        vals = [board[r][c] for r, c in line]
        if vals[0] != EMPTY and all(v == vals[0] for v in vals):
            return vals[0]
    return None


def _move_creates_win(board, r, c):
    """True if the mark at (r, c) completes one of its winning lines."""
    mark = board[r][c]
    for line in CELL_LINES[(r, c)]:
        if all(board[rr][cc] == mark for rr, cc in line):
            return True
    return False


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
# Strong built-in heuristic: depth-aware minimax with alpha-beta pruning,
# a static line-based evaluator for the early/mid game, and variable
# accuracy (probability of playing the optimal move).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Improved static evaluator
# ---------------------------------------------------------------------------
# Three enhancements over the original "live-lines" approach:
#
#  1. Open-end awareness  (scribd.com / NTU CS3243 lecture notes)
#     A run of k marks is far more dangerous when BOTH ends of its line are
#     still open (the opponent cannot block it with a single move).  We
#     award 2× the base weight for fully open lines vs 1× for half-open.
#
#  2. Fork detection bonus  (researchgate.net / GeeksForGeeks minimax)
#     A «fork» is having two or more simultaneous k=3 threats.  It is the
#     single most decisive tactical motif in 4×4 Tic-Tac-Toe because the
#     opponent can only block one of them.  We add a flat bonus whenever a
#     player holds ≥2 live three-in-a-row lines simultaneously.
#
#  3. Positional (cell-participation) bonuses  (trincoll.edu CS notes)
#     Each cell's value is proportional to how many winning lines run
#     through it (stored in CELL_WEIGHTS).  Placing marks on high-
#     participation cells keeps more future winning options alive.
# ---------------------------------------------------------------------------

# Base weights indexed by run length (1 / 2 / 3 marks in an unblocked line)
_RUN_WEIGHTS = {1: 1, 2: 10, 3: 100}
_FORK_BONUS   = 500   # extra score for holding ≥2 simultaneous 3-threats
_CELL_BONUS   = 0.5   # positional bonus per unit of CELL_WEIGHTS


def _count_open_ends(vals):
    """Return how many ends of a 4-cell line are still empty.

    For a line represented as a flat list of 4 cell values, the two
    'ends' that can extend a run are simply the first and last elements.
    An end is open when the cell at that position is EMPTY.
    """
    return (1 if vals[0] == EMPTY else 0) + (1 if vals[-1] == EMPTY else 0)


def _evaluate_position(board, maximizing_player):
    """
    Improved static evaluation used when the depth-limited search must
    score a non-terminal board.  Combines three complementary signals:

    1. Weighted run-length score with open-end distinction
       (2× weight when both ends of the line are open, 1× otherwise).
    2. Fork bonus for holding two or more simultaneous 3-in-a-row threats.
    3. Positional bonus proportional to how many winning lines pass through
       each occupied cell (CELL_WEIGHTS).
    """
    minimizing_player = other_player(maximizing_player)
    score = 0

    # --- Signal 1 + 2: run-length scoring with open-end awareness ----------
    max_threats = 0   # number of live 3-in-a-row lines for maximizing player
    min_threats = 0

    for line in WINNING_LINES:
        vals = [board[r][c] for r, c in line]
        max_count = vals.count(maximizing_player)
        min_count = vals.count(minimizing_player)

        if max_count and min_count:
            continue  # mutually blocked: dead line, worth nothing

        open_ends = _count_open_ends(vals)
        # A line with zero open ends and <4 marks can never be completed.
        if open_ends == 0 and max_count < BOARD_SIZE and min_count < BOARD_SIZE:
            continue

        # Multiplier: fully open lines are twice as valuable.
        openness = 2 if open_ends == 2 else 1

        if max_count:
            base = _RUN_WEIGHTS.get(max_count, 0)
            score += base * openness
            if max_count == 3:
                max_threats += 1
        elif min_count:
            base = _RUN_WEIGHTS.get(min_count, 0)
            score -= base * openness
            if min_count == 3:
                min_threats += 1

    # --- Signal 2: fork bonus -----------------------------------------------
    if max_threats >= 2:
        score += _FORK_BONUS
    if min_threats >= 2:
        score -= _FORK_BONUS

    # --- Signal 3: positional bonus -----------------------------------------
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = board[r][c]
            if cell == maximizing_player:
                score += CELL_WEIGHTS[(r, c)] * _CELL_BONUS
            elif cell == minimizing_player:
                score -= CELL_WEIGHTS[(r, c)] * _CELL_BONUS

    return score


def _minimax(board, player, maximizing_player, depth, last_move=None,
             alpha=-float('inf'), beta=float('inf')):
    """
    Returns (score, move) for `player` to move, from the perspective of
    `maximizing_player`. Positive scores favor maximizing_player winning,
    negative favor the opponent, 0 is a draw. `depth` is the remaining
    ply budget; once it hits 0 without the game having ended, falls back
    to the static evaluator instead of continuing to search.

    `last_move` is the (row, col) just played to reach this board (None
    at the root). Since the caller guarantees the root board is not
    already a terminal position, the only way a *new* win can appear is
    through the line(s) that pass through the last move, so that's all
    that needs checking here -- much cheaper than rescanning all 10
    winning lines at every node.
    """
    if last_move is not None:
        lr, lc = last_move
        if _move_creates_win(board, lr, lc):
            mover = board[lr][lc]
            if mover == maximizing_player:
                return 1_000_000 + depth, None   # prefer faster wins
            return -1_000_000 - depth, None      # prefer slower losses

    moves = get_valid_moves(board)
    if not moves:
        return 0, None  # board full, no winner -> draw
    if depth == 0:
        return _evaluate_position(board, maximizing_player), None

    # Move ordering: sort moves by a one-ply static eval before expanding.
    # Examining the most promising moves first maximises alpha-beta cutoffs,
    # effectively roughly doubling the search depth for the same node budget.
    # Source: stackoverflow.com/questions/tagged/minimax+alpha-beta
    def _quick_score(move):
        r, c = move
        board[r][c] = player
        s = _evaluate_position(board, maximizing_player)
        board[r][c] = EMPTY
        return s

    moves = sorted(moves, key=_quick_score,
                   reverse=(player == maximizing_player))

    best_move = moves[0]

    if player == maximizing_player:
        best_score = -float('inf')
        for move in moves:
            r, c = move
            board[r][c] = player
            score, _ = _minimax(board, other_player(player), maximizing_player, depth - 1, move, alpha, beta)
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
            score, _ = _minimax(board, other_player(player), maximizing_player, depth - 1, move, alpha, beta)
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

    The random coin is flipped *before* running the search, so the
    (relatively expensive) minimax search is skipped whenever its result
    would be discarded anyway.
    """
    def best_heuristic(board):
        moves = get_valid_moves(board)
        if not moves:
            return None

        # With probability (1 - accuracy), play a random move immediately
        # without running the expensive search.
        if random.random() >= accuracy:
            return random.choice(moves)

        empty_cells = len(moves)
        depth = empty_cells if empty_cells <= FULL_SEARCH_THRESHOLD else LOOKAHEAD_DEPTH

        _, move = _minimax(copy.deepcopy(board), PLAYER_Y, PLAYER_Y, depth)
        if move is None:
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
