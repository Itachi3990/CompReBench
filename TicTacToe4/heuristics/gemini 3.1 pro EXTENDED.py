import math

def myHeuristic(board):
    """
    Determines the best move for 'X' on a 4x4 Tic-Tac-Toe board.
    
    :param board: 2D list of strings representing the board state (4x4).
                  'X' is the AI, 'Y' is the opponent, '' is an empty cell.
    :return: A tuple of integers (row, col) representing the best move.
    """
    empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r][c] == '']

    # Edge case: If the board is full, no moves can be made
    if not empty_cells:
        return None

    # Depth determination based on branching factor
    if len(empty_cells) > 8:
        max_depth = 4
    else:
        max_depth = len(empty_cells)

    best_score = -math.inf
    best_move = empty_cells[0]
    alpha = -math.inf
    beta = math.inf

    # Iterate through all valid moves to find the best one
    for r, c in empty_cells:
        board[r][c] = 'X'
        score = _minimax(board, max_depth - 1, alpha, beta, False)
        board[r][c] = ''  # Undo move

        if score > best_score:
            best_score = score
            best_move = (r, c)

        alpha = max(alpha, best_score)

    return best_move

def _minimax(board, depth, alpha, beta, is_maximizing):
    """
    Minimax algorithm with Alpha-Beta pruning.
    """
    winner = _check_winner(board)
    
    # Base cases: Terminal states (Win/Lose/Draw)
    # We add or subtract the depth to reward faster wins and delay inevitable losses.
    if winner == 'X':
        return 10000 + depth
    if winner == 'Y':
        return -10000 - depth
    if winner == 'Draw':
        return 0

    # Base case: Depth limit reached
    if depth == 0:
        return _evaluate(board)

    empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r][c] == '']

    if is_maximizing:
        max_eval = -math.inf
        for r, c in empty_cells:
            board[r][c] = 'X'
            eval_score = _minimax(board, depth - 1, alpha, beta, False)
            board[r][c] = ''
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for r, c in empty_cells:
            board[r][c] = 'Y'
            eval_score = _minimax(board, depth - 1, alpha, beta, True)
            board[r][c] = ''
            
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def _check_winner(board):
    """
    Checks if there is a winner or a draw on the board.
    Returns 'X', 'Y', 'Draw', or None.
    """
    # Check rows
    for r in range(4):
        if board[r][0] == board[r][1] == board[r][2] == board[r][3] and board[r][0] != '':
            return board[r][0]
            
    # Check columns
    for c in range(4):
        if board[0][c] == board[1][c] == board[2][c] == board[3][c] and board[0][c] != '':
            return board[0][c]
            
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] == board[3][3] and board[0][0] != '':
        return board[0][0]
    if board[0][3] == board[1][2] == board[2][1] == board[3][0] and board[0][3] != '':
        return board[0][3]

    # Check for draw
    for r in range(4):
        for c in range(4):
            if board[r][c] == '':
                return None
                
    return 'Draw'

def _evaluate(board):
    """
    Heuristic evaluation function.
    Calculates a score based on the number of potential winning lines for 'X' and 'Y'.
    """
    score = 0
    lines = []
    
    # Collect all possible winning lines
    for r in range(4):
        lines.append([board[r][c] for c in range(4)])
    for c in range(4):
        lines.append([board[r][c] for r in range(4)])
        
    lines.append([board[i][i] for i in range(4)])
    lines.append([board[i][3-i] for i in range(4)])

    # Evaluate each line
    for line in lines:
        x_count = line.count('X')
        y_count = line.count('Y')

        # If line has only X's, it's a potential win for X
        if x_count > 0 and y_count == 0:
            if x_count == 1:
                score += 1
            elif x_count == 2:
                score += 10
            elif x_count == 3:
                score += 100
                
        # If line has only Y's, it's a potential win for Y
        elif y_count > 0 and x_count == 0:
            if y_count == 1:
                score -= 1
            elif y_count == 2:
                score -= 10
            elif y_count == 3:
                score -= 100

    return score