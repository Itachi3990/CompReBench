def check_winner(board):
    """
    Checks if there is a winner or if the game ended in a draw.
    Returns: 'X', 'Y', 'Draw', or None (if the game is still ongoing).
    """
    # Check rows and columns
    for i in range(4):
        if board[i][0] != '' and board[i][0] == board[i][1] == board[i][2] == board[i][3]:
            return board[i][0]
        if board[0][i] != '' and board[0][i] == board[1][i] == board[2][i] == board[3][i]:
            return board[0][i]

    # Check main and anti diagonals
    if board[0][0] != '' and board[0][0] == board[1][1] == board[2][2] == board[3][3]:
        return board[0][0]
    if board[0][3] != '' and board[0][3] == board[1][2] == board[2][1] == board[3][0]:
        return board[0][3]

    # Check if any empty cells remain
    for r in range(4):
        for c in range(4):
            if board[r][c] == '' or board[r][c] == ' ':
                return None  # Game ongoing

    return 'Draw'


def evaluate_board_state(board):
    """
    Helper function to evaluate static non-terminal board states.
    Scores line combinations for 'X' (positive) vs 'Y' (negative).
    """
    def score_line(line):
        x_count = line.count('X')
        y_count = line.count('Y')

        # If a line contains both players' marks, neither can win on it
        if x_count > 0 and y_count > 0:
            return 0
        if x_count == 4:
            return 10000
        if y_count == 4:
            return -10000
        if x_count == 3:
            return 100
        if x_count == 2:
            return 10
        if x_count == 1:
            return 1
        if y_count == 3:
            return -100
        if y_count == 2:
            return -10
        if y_count == 1:
            return -1
        return 0

    lines = []
    # Collect 4 rows and 4 columns
    for i in range(4):
        lines.append([board[i][c] for c in range(4)])
        lines.append([board[r][i] for r in range(4)])

    # Collect diagonals
    lines.append([board[i][i] for i in range(4)])
    lines.append([board[i][3 - i] for i in range(4)])

    return sum(score_line(line) for line in lines)


def _get_empty_cells(board):
    """Returns a list of tuples (row, col) for all empty board positions."""
    empty = []
    for r in range(4):
        for c in range(4):
            if board[r][c] == '' or board[r][c] == ' ':
                empty.append((r, c))
    
    # Sort empty cells by proximity to center to optimize alpha-beta pruning
    empty.sort(key=lambda pos: abs(pos[0] - 1.5) + abs(pos[1] - 1.5))
    return empty


def _alpha_beta(board, depth, alpha, beta, is_maximizing):
    """
    Minimax search with Alpha-Beta Pruning.
    'X' is the maximizing player; 'Y' is the minimizing player.
    """
    winner = check_winner(board)
    
    # Terminal utility scores
    if winner == 'X':
        return 10000 + depth, None  # Prefer winning sooner
    elif winner == 'Y':
        return -10000 - depth, None # Prefer losing later
    elif winner == 'Draw':
        return 0, None

    # Depth limit reached: use heuristic helper evaluation
    if depth == 0:
        return evaluate_board_state(board), None

    empty_cells = _get_empty_cells(board)
    if not empty_cells:
        return 0, None

    best_move = empty_cells[0]

    if is_maximizing:
        max_eval = -float('inf')
        for r, c in empty_cells:
            board[r][c] = 'X'
            eval_score, _ = _alpha_beta(board, depth - 1, alpha, beta, False)
            board[r][c] = ''

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = (r, c)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for r, c in empty_cells:
            board[r][c] = 'Y'
            eval_score, _ = _alpha_beta(board, depth - 1, alpha, beta, True)
            board[r][c] = ''

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = (r, c)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval, best_move


def myHeuristic(board):
    """
    Main entry point function required by the external game runner.
    
    :param board: 2D list (4x4) representing current state with 'X', 'Y', or ''
    :return: Tuple (row, col) representing the best move for 'X'
    """
    empty_cells = _get_empty_cells(board)
    if not empty_cells:
        return None

    num_empty = len(empty_cells)

    # Dynamic depth adjustment as specified:
    # Look ahead 4 levels if > 8 empty cells; otherwise search to terminal nodes.
    max_depth = 4 if num_empty > 8 else num_empty

    _, best_move = _alpha_beta(
        board=board,
        depth=max_depth,
        alpha=-float('inf'),
        beta=float('inf'),
        is_maximizing=True
    )

    return best_move