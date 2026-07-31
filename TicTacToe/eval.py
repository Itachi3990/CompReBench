"""
Tic-Tac-Toe Heuristic Evaluator
================================

This script batch-evaluates a folder of externally-supplied heuristic
functions against a strong built-in heuristic, 500 games each, and prints
each test taker's score.

BOARD INTERFACE CONTRACT
-------------------------
A board is a 3x3 list of lists (rows, then columns), e.g.:

    board[0] = [row0col0, row0col1, row0col2]
    board[1] = [row1col0, row1col1, row1col2]
    board[2] = [row2col0, row2col1, row2col2]

Each cell holds one of:
    ''   -> empty cell
    'X'  -> X has been placed there
    'O'  -> O has been placed there

X always moves first. Whose turn it is can be inferred from the board:
if the number of X's equals the number of O's, it is X's turn, otherwise
it is O's turn. A heuristic function does not need to be told which side
it is playing -- it is always expected to produce the best move for
whichever side is currently "to move" on the given board.

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
For each test taker's file, 500 games are played against the built-in
heuristic (alternating who moves first). Their score is:

    score = (number of wins + draws) * 0.2

OUTPUT
------
For each heuristic file, a single-line progress bar is shown while its
500 games are played. Once every file has been evaluated, a final table
of "name: score" is printed to the console.
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
PLAYER_O = 'O'

TOTAL_MATCHES = 500
SCORE_MULTIPLIER = 0.2
HEURISTICS_DIR = 'heuristics'


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


def current_player(board):
    x_count = sum(row.count(PLAYER_X) for row in board)
    o_count = sum(row.count(PLAYER_O) for row in board)
    return PLAYER_X if x_count == o_count else PLAYER_O


def other_player(p):
    return PLAYER_O if p == PLAYER_X else PLAYER_X


# ---------------------------------------------------------------------------
# Strong built-in heuristic: minimax with alpha-beta pruning + 50% randomness
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


def best_heuristic(board):
    """
    Computes the game-theoretically optimal move via alpha-beta minimax,
    then, just before returning, has a 50% chance of substituting a
    uniformly random legal move instead.
    """
    moves = get_valid_moves(board)
    if not moves:
        return None

    player = current_player(board)
    _, move = _minimax(copy.deepcopy(board), player, player)
    if move is None:
        move = random.choice(moves)

    if random.random() < 0.5:
        move = random.choice(moves)

    return move


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


def play_match(x_heuristic, o_heuristic):
    """
    Plays a single game. x_heuristic controls X, o_heuristic controls O.
    Returns 'X', 'O', or 'draw'.
    A player that returns an invalid move forfeits the match.
    """
    board = create_board()
    controllers = {PLAYER_X: x_heuristic, PLAYER_O: o_heuristic}

    while True:
        winner = check_winner(board)
        if winner is not None:
            return winner
        if is_draw(board):
            return 'draw'

        player = current_player(board)
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
    """Plays TOTAL_MATCHES games of heuristic_func vs. best_heuristic."""
    wins_or_draws = 0

    for i in range(1, TOTAL_MATCHES + 1):
        # Alternate which side the test taker's heuristic plays, so
        # neither side is systematically favored by moving first.
        if i % 2 == 1:
            result = play_match(heuristic_func, best_heuristic)
            their_symbol = PLAYER_X
        else:
            result = play_match(best_heuristic, heuristic_func)
            their_symbol = PLAYER_O

        if result == 'draw' or result == their_symbol:
            wins_or_draws += 1

        print_progress(name, i, TOTAL_MATCHES)

    return wins_or_draws


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

        wins_or_draws = evaluate_heuristic(name, heuristic_func)
        score = round(wins_or_draws * SCORE_MULTIPLIER, 1)
        results.append((name, score))

    print()
    for name, score in results:
        print(f"{name}: {score}")


if __name__ == '__main__':
    main()
